import io
import zipfile

import fitz
from fastapi.testclient import TestClient

from app import config
from app.api import app

c = TestClient(app)


def _zip_with_pdf():
    d = fitz.open()
    d.new_page().insert_text((72, 72), "Acme Corp consolidated revenue was 10 million in FY2023.")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.pdf", d.tobytes())
        z.writestr("copy.pdf", d.tobytes())  # exact duplicate
    return buf.getvalue()


def test_index_then_query_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STORAGE_DIR", tmp_path)
    r = c.post("/index", files={"file": ("d.zip", _zip_with_pdf())}, data={"collection_name": "demo"})
    assert r.status_code == 200, r.text
    assert r.json() == {"status": "success", "collection_name": "demo"}
    assert (tmp_path / "demo" / "dense.faiss").exists()
    q = c.post("/query", json={"collection_name": "demo", "query": "hi"}).json()
    assert set(q) == {"answer", "chunks", "trace"}
    for step in q["trace"]:
        assert set(step) == {"step_name", "description", "input", "output"}


def test_invalid_zip_is_400(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STORAGE_DIR", tmp_path)
    r = c.post("/index", files={"file": ("d.zip", b"nope")}, data={"collection_name": "x"})
    assert r.status_code == 400


def test_bad_collection_names_rejected():
    assert c.post("/query", json={"collection_name": "../x", "query": "hi"}).status_code == 400
