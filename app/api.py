import logging
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from . import config, indexer
from .schemas import QueryRequest, QueryResponse, TraceStep

log = logging.getLogger("verityx")
app = FastAPI(title="VERITYX")
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
async def query(req: QueryRequest):
    d = collection_dir(req.collection_name)
    if not d.exists():
        raise HTTPException(404, "unknown collection")
    # TODO Phase B: contract -> retrieval -> verify/prune -> answer
    trace = [TraceStep(step_name="stub", description="skeleton response",
                       input={"query": req.query}, output={})]
    return QueryResponse(answer="", chunks=[], trace=trace)
