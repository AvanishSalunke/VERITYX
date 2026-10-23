"""Phase A orchestration: zip -> profiles/gists/chunks -> SQLite + FAISS + BM25."""
import json
import re
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import faiss
import numpy as np

from . import config, db, ingest, profile
from .tracing import observe

PROFILE_WORKERS = 16
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
_locks: dict[str, threading.Lock] = {}
_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding(EMBED_MODEL)
    return _embedder


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", text.lower())


def embed(texts: list[str]) -> np.ndarray:
    v = np.array(list(get_embedder().embed(texts)), dtype="float32")
    faiss.normalize_L2(v)
    return v


def context_prefix(prof: dict, doc_label: str, page: int) -> str:
    ents = ", ".join(prof.get("entities") or [])[:120]
    per = ", ".join(prof.get("reporting_periods") or [])[:80]
    return (f"[Doc: {doc_label} | Entity: {ents} | Period: {per} | Scope: {prof.get('scope')} "
            f"| Page: {page}] ")


@observe(name="index_collection")
def build_collection(zip_path: Path, collection: str) -> dict:
    lock = _locks.setdefault(collection, threading.Lock())
    if not lock.acquire(blocking=False):
        raise RuntimeError("collection is already being indexed")
    work = Path(tempfile.mkdtemp(prefix="verityx_"))
    stage = config.STORAGE_DIR / f".{collection}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        pdfs = ingest.safe_unzip(zip_path, work)
        if not pdfs:
            raise ValueError("no PDFs found in archive")
        groups = ingest.dedupe(pdfs)
        con = db.connect(stage)
        ctx_texts, chunk_ids = [], []
        stats = {"documents": 0, "duplicates": 0, "pages": 0, "low_text_pages": 0, "chunks": 0}
        docs = []  # (doc_id, sha, names, pages)
        reps = [paths[0] for paths in groups.values()]
        for (sha, paths), pages in zip(groups.items(), ingest.extract_many(reps)):
            stats["duplicates"] += len(paths) - 1
            if isinstance(pages, Exception):  # corrupt PDF: skip, keep going
                stats.setdefault("failed", []).append(f"{paths[0].name}: {str(pages)[:80]}")
                continue
            docs.append(("doc_" + sha[:10], sha, sorted({p.name for p in paths}), pages))

        def make_profile(d):
            doc_id, _, _, pages = d
            det = profile.deterministic_facts("\n".join(p.text for p in pages))
            try:
                return profile.build_profile(doc_id, "\n\n".join(p.text for p in pages[:4]), det)
            except Exception:
                return {"doc_id": doc_id, "entities": [], "reporting_periods": det["fiscal_periods"],
                        "scope": None, "doc_type": None, "summary": "", "deterministic": det}

        with ThreadPoolExecutor(PROFILE_WORKERS) as ex:
            profiles = list(ex.map(make_profile, docs))

        for (doc_id, sha, names, pages), prof in zip(docs, profiles):
            con.execute("INSERT INTO documents VALUES(?,?,?,?,?)",
                        (doc_id, sha, json.dumps(names), len(pages), json.dumps(prof)))
            for page in pages:
                con.execute("INSERT INTO pages VALUES(?,?,?,?,?,?,?)",
                            (doc_id, page.pdf_page, page.label, page.text, int(page.low_text), "", "{}"))
                stats["pages"] += 1
                stats["low_text_pages"] += int(page.low_text)
                for kind, text in ingest.chunk_page(page):
                    cur = con.execute("INSERT INTO chunks(doc_id,pdf_page,kind,text) VALUES(?,?,?,?)",
                                      (doc_id, page.pdf_page, kind, text))
                    chunk_ids.append(cur.lastrowid)
                    ctx_texts.append(context_prefix(prof, names[0], page.pdf_page) + text)
            stats["documents"] += 1
        if not chunk_ids:
            raise ValueError("no extractable text found")
        con.commit()
        vecs = embed(ctx_texts)
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        faiss.write_index(index, str(stage / "dense.faiss"))
        (stage / "bm25_tokens.json").write_text(
            json.dumps({"chunk_ids": chunk_ids, "tokens": [tokenize(t) for t in ctx_texts]}))
        stats["chunks"] = len(chunk_ids)
        con.close()
        final = config.STORAGE_DIR / collection
        if final.exists():
            shutil.rmtree(final)
        stage.rename(final)
        return stats
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
        lock.release()
