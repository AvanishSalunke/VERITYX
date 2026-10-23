import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(
  doc_id TEXT PRIMARY KEY, sha256 TEXT UNIQUE, filenames TEXT, n_pages INTEGER,
  profile TEXT);
CREATE TABLE IF NOT EXISTS pages(
  doc_id TEXT, pdf_page INTEGER, label TEXT, text TEXT, low_text INTEGER,
  gist TEXT, catalogue TEXT, PRIMARY KEY(doc_id, pdf_page));
CREATE TABLE IF NOT EXISTS chunks(
  chunk_id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id TEXT, pdf_page INTEGER,
  kind TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS relationships(
  src TEXT, rel TEXT, dst TEXT, doc_id TEXT, pdf_page INTEGER);
"""


def connect(collection_dir: Path) -> sqlite3.Connection:
    collection_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(collection_dir / "meta.db")
    con.executescript(SCHEMA)
    return con
