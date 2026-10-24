"""HTTP API: exactly two endpoints.

    POST /index   multipart: file=<zip of documents>, collection_name=<name>
    POST /query   JSON: {"collection_name": "...", "query": "..."}

    uvicorn verityx.api:app --host 0.0.0.0 --port 8000

Each collection lives in its own storage/<collection_name>/ (documents, chunks, Doc Cards,
embeddings), so indexing a new corpus never touches another collection. FastAPI's generated
/docs and /openapi.json routes are disabled so only the two required endpoints are exposed.
"""
import json
import re
import shutil
import threading
import time
import zipfile
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from . import tracing
from .index import build_collection, collection_dir, load_collection
from .pipeline import run_query
from .planner import Planner
from .retrieve import Retriever

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
FILE_NAMES = "file_names.json"   # stored doc name -> file name as given in the zip
TRACE_KEYS = ("step_name", "description", "input", "output")

app = FastAPI(title="VERITYX", docs_url=None, redoc_url=None, openapi_url=None)

_engines = {}                         # collection -> (Retriever, Planner)
_locks = defaultdict(threading.Lock)  # one indexing/loading lock per collection
_guard = threading.Lock()


def _lock(name):
    with _guard:
        return _locks[name]


def _check_name(name):
    if not NAME.match(name or ""):
        raise HTTPException(400, "collection_name must be 1-64 chars of letters, digits, _ . -")


def _extract(zip_path, dest):
    """Unzip PDFs into a flat folder. Guards against path traversal, skips macOS metadata and
    disambiguates files with the same name in different folders. Returns {stored name: name in
    the zip}, so responses can report each document's file name exactly as it was provided."""
    dest.mkdir(parents=True)
    names = {}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            parts = Path(info.filename).parts
            if (info.is_dir() or "__MACOSX" in parts or parts[-1].startswith("._")
                    or not parts[-1].lower().endswith(".pdf")):
                continue
            if Path(info.filename).is_absolute() or ".." in parts:
                continue
            name = parts[-1]
            if (dest / name).exists():
                name = "_".join(parts[-2:]) if len(parts) > 1 else f"{len(names)}_{name}"
            with zf.open(info) as src, open(dest / name, "wb") as out:
                shutil.copyfileobj(src, out)
            names[name] = parts[-1]
    return names


def _engine(name):
    """(Retriever, Planner, {stored doc name: original file name}) for a collection."""
    if name in _engines:
        return _engines[name]
    with _lock(name):
        if name not in _engines:
            if not (collection_dir(name) / "chunks.jsonl").exists():
                raise HTTPException(404, "unknown collection")
            chunks, cards, emb = load_collection(name)
            names_file = collection_dir(name) / FILE_NAMES
            names = (json.loads(names_file.read_text(encoding="utf-8"))
                     if names_file.exists() else {})
            _engines[name] = (Retriever(chunks, cards, emb), Planner(cards), names)
    return _engines[name]


@app.post("/index")
def index(file: UploadFile = File(...), collection_name: str = Form(...)):
    _check_name(collection_name)
    t0 = time.time()
    with _lock(collection_name):
        base = collection_dir(collection_name)
        base.mkdir(parents=True, exist_ok=True)
        upload = base / "upload.zip"
        with open(upload, "wb") as out:
            shutil.copyfileobj(file.file, out)
        docs = base / "docs"
        if docs.exists():
            shutil.rmtree(docs)
        try:
            names = _extract(upload, docs)
            n_docs = len(names)
            if not n_docs:
                raise HTTPException(400, "the zip contains no PDF documents")
        except (zipfile.BadZipFile, HTTPException) as e:
            if not (base / "chunks.jsonl").exists():   # never indexed: leave nothing behind
                shutil.rmtree(base, ignore_errors=True)
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(400, "file is not a valid zip archive")
        finally:
            upload.unlink(missing_ok=True)
        stats = build_collection(docs, collection_name)
        (base / FILE_NAMES).write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
        _engines.pop(collection_name, None)      # next query loads the new index
    return {"status": "success", "collection_name": collection_name,
            "documents_indexed": n_docs, "chunks": stats["chunks"],
            "ocr_pages": stats.get("ocr_pages", 0),
            "indexing_time_s": round(time.time() - t0, 1)}


class QueryIn(BaseModel):
    collection_name: str
    query: str


@app.post("/query")
def query(body: QueryIn):
    if not NAME.match(body.collection_name or ""):
        raise HTTPException(404, "unknown collection")    # no such collection can exist
    if not body.query.strip():
        raise HTTPException(400, "query is empty")
    retriever, planner, names = _engine(body.collection_name)
    r = run_query(retriever, planner, body.query, collection=body.collection_name)
    tracing.flush()
    if any(s["step_name"] == "llm_answer_generation" and s["description"] == "failed"
           for s in r["trace"]):
        # Contract: internal errors are 5xx so the harness retries; a 200 with no answer
        # would be scored as zero.
        raise HTTPException(503, "answer generation failed; please retry")
    cited = {id(c) for c in r["cited"]}
    # An answer that cites nothing is a "cannot answer": the contract expects empty chunks,
    # and any chunk returned would only lower context precision.
    used = r["chunks"] if r["cited"] else []
    return {
        "answer": r["answer"],
        # Contract shape: content + metadata.file_name (as in the zip) + metadata.page_numbers
        # (1-based). Extra metadata keys are allowed and ignored by the harness.
        "chunks": [{"content": c["text"],
                    "metadata": {"file_name": names.get(c["doc"], c["doc"]),
                                 "page_numbers": [c["page"]],
                                 "section": c["section"], "chunk_id": c["id"],
                                 "cited_in_answer": id(c) in cited}}
                   for c in used],
        "trace": [{k: s.get(k) for k in TRACE_KEYS} for s in r["trace"]],
        "latency_s": r["latency_s"],
        "cost_usd": r["cost_usd"],
        "tokens": r["tokens"],
        "langfuse_trace_url": r["langfuse_trace_url"],
    }
