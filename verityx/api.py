"""HTTP API: exactly two endpoints.

    POST /index   multipart: file=<zip of documents>, collection_name=<name>
    POST /query   JSON: {"collection_name": "...", "query": "..."}

    uvicorn verityx.api:app --host 0.0.0.0 --port 8000

Each collection lives in its own storage/<collection_name>/ (documents, chunks, Doc Cards,
embeddings), so indexing a new corpus never touches another collection. FastAPI's generated
/docs and /openapi.json routes are disabled so only the two required endpoints are exposed.
"""
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
    disambiguates files with the same name in different folders."""
    dest.mkdir(parents=True)
    n = 0
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
                name = "_".join(parts[-2:]) if len(parts) > 1 else f"{n}_{name}"
            with zf.open(info) as src, open(dest / name, "wb") as out:
                shutil.copyfileobj(src, out)
            n += 1
    return n


def _engine(name):
    if name in _engines:
        return _engines[name]
    with _lock(name):
        if name not in _engines:
            if not (collection_dir(name) / "chunks.jsonl").exists():
                raise HTTPException(404, f"collection '{name}' is not indexed")
            chunks, cards, emb = load_collection(name)
            _engines[name] = (Retriever(chunks, cards, emb), Planner(cards))
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
            n_docs = _extract(upload, docs)
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
        _engines.pop(collection_name, None)      # next query loads the new index
    return {"status": "success", "collection_name": collection_name,
            "documents": n_docs, "chunks": stats["chunks"],
            "ocr_pages": stats.get("ocr_pages", 0),
            "indexing_time_s": round(time.time() - t0, 1)}


class QueryIn(BaseModel):
    collection_name: str
    query: str


@app.post("/query")
def query(body: QueryIn):
    _check_name(body.collection_name)
    if not body.query.strip():
        raise HTTPException(400, "query is empty")
    retriever, planner = _engine(body.collection_name)
    r = run_query(retriever, planner, body.query, collection=body.collection_name)
    tracing.flush()
    cited = {id(c) for c in r["cited"]}
    return {
        "answer": r["answer"],
        "chunks": [{"chunk_id": c["id"], "document": c["doc"], "page": c["page"],
                    "page_numbers": [c["page"]], "section": c["section"], "text": c["text"],
                    "cited_in_answer": id(c) in cited} for c in r["chunks"]],
        "trace": [{k: s.get(k) for k in TRACE_KEYS} for s in r["trace"]],
        "latency_s": r["latency_s"],
        "cost_usd": r["cost_usd"],
        "tokens": r["tokens"],
        "langfuse_trace_url": r["langfuse_trace_url"],
    }
