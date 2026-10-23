"""Phase A, step 1: safe unzip, SHA-256 dedupe, page-preserving extraction, chunking."""
import hashlib
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import fitz

MIN_CHARS_FOR_TEXT = 40  # below this a page is treated as scanned / empty
MAX_CHUNK_WORDS = 180


@dataclass
class PageRec:
    pdf_page: int          # 1-based physical page
    label: str             # printed page label, "" if none
    text: str
    low_text: bool
    tables: list = field(default_factory=list)  # list[list[list[str]]]


@dataclass
class DocRec:
    sha256: str
    filenames: list
    pages: list


def safe_unzip(zip_path: Path, dest: Path, max_bytes: int = 4 << 30) -> list[Path]:
    """Extract PDFs only; reject path traversal and oversized archives."""
    out, total = [], 0
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".pdf"):
                continue
            target = (dest / info.filename).resolve()
            if dest not in target.parents:
                raise ValueError(f"unsafe path in archive: {info.filename}")
            total += info.file_size
            if total > max_bytes:
                raise ValueError("archive too large")
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            out.append(target)
    return out


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dedupe(paths: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for p in paths:
        groups.setdefault(sha256_file(p), []).append(p)
    return groups


def extract_pdf(path: Path) -> list[PageRec]:
    pages = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text") or ""
            tables = []
            try:
                for t in page.find_tables().tables:
                    rows = [[(c or "").strip() for c in r] for r in t.extract()]
                    if rows:
                        tables.append(rows)
            except Exception:
                pass  # table detection is best-effort
            pages.append(PageRec(i + 1, page.get_label() or "", text,
                                 len(text.strip()) < MIN_CHARS_FOR_TEXT, tables))
    return pages


def table_to_text(rows: list[list[str]]) -> str:
    return "\n".join(" | ".join(r) for r in rows)


def chunk_page(page: PageRec) -> list[tuple[str, str]]:
    """Return (kind, text) chunks. Never crosses a page boundary."""
    chunks = []
    for t in page.tables:
        txt = table_to_text(t)
        if txt.strip():
            chunks.append(("table", txt))
    paras = [p.strip() for p in re.split(r"\n\s*\n", page.text) if p.strip()]
    buf: list[str] = []
    n = 0
    for p in paras:
        w = len(p.split())
        if buf and n + w > MAX_CHUNK_WORDS:
            chunks.append(("text", "\n".join(buf)))
            buf, n = [], 0
        buf.append(p)
        n += w
    if buf:
        chunks.append(("text", "\n".join(buf)))
    return chunks
