import logging
import re
import shutil
import tempfile
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from . import answer as answer_mod
from . import config, indexer, retrieve
from .contract import compile_contract
from .schemas import Chunk, QueryRequest, QueryResponse, TraceStep

log = logging.getLogger("verityx")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the embedder and existing collections up front so the first query isn't slow."""
    indexer.embed(["warmup"])
    if config.STORAGE_DIR.exists():
        for d in config.STORAGE_DIR.iterdir():
            if (d / "meta.db").exists() and _NAME.match(d.name):
                try:
                    retrieve.load_collection(d.name)
                except Exception as e:  # noqa: BLE001 - a broken collection must not block startup
                    log.warning("could not preload %s: %s", d.name, e)
    yield


app = FastAPI(title="VERITYX", lifespan=lifespan)
_NAME = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def collection_dir(name: str) -> Path:
    if not _NAME.match(name):
        raise HTTPException(400, "invalid collection_name")
    return config.STORAGE_DIR / name


@app.post("/index")
def index(file: UploadFile = File(...), collection_name: str = Form(...)):
    collection_dir(collection_name)  # validates the name
    config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
    zpath = Path(tmp.name)
    try:
        stats = indexer.build_collection(zpath, collection_name)
    except zipfile.BadZipFile:
        raise HTTPException(400, "not a valid zip archive")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    finally:
        zpath.unlink(missing_ok=True)
    log.info("indexed %s: %s", collection_name, stats)
    return {"status": "success", "collection_name": collection_name}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    d = collection_dir(req.collection_name)
    if not (d / "meta.db").exists():
        raise HTTPException(404, "unknown collection")
    col = retrieve.load_collection(req.collection_name)
    trace = []
    docs = col.docs()
    contract = compile_contract(req.query, docs)
    trace.append(TraceStep(
        step_name="compile_evidence_contract",
        description="Question decomposed into typed requirements and candidate documents",
        input={"query": req.query, "n_documents_in_manifest": len(docs)},
        output=contract.model_dump()))
    res = retrieve.execute(col, contract)
    trace.append(TraceStep(
        step_name="execute_contract_retrieval",
        description="Dense + BM25 retrieval per requirement, restricted to target documents",
        input={"requirements": [r.id for r in contract.requirements]},
        output={"per_requirement_chunk_ids": res["per_requirement"], "n_chunks": len(res["chunks"])}))
    answer = answer_mod.generate(req.query, contract, res["chunks"])
    trace.append(TraceStep(
        step_name="generate_grounded_answer",
        description="Cited answer generated only from the selected chunks",
        input={"n_chunks": len(res["chunks"])},
        output={"answer_chars": len(answer)}))
    chunks = [Chunk(text=c["text"], document=c["document"], page=c["page"]) for c in res["chunks"]]
    return QueryResponse(answer=answer, chunks=chunks, trace=trace)
