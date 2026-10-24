"""OCR fallback for pages with no text layer (scanned PDFs).

Only pages whose extracted text is (nearly) empty are sent, as page images, to a vision model
on the same OpenCode key; everything else keeps PyMuPDF's text. Local OCR (RapidOCR) measured
16-27 s/page on this machine, versus ~6 s/page here with pages run in parallel. Results are
cached per (file hash, page), so re-indexing a collection does not re-OCR.
"""
import base64
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf

from . import tracing

CACHE_DIR = Path(__file__).resolve().parents[1] / "storage" / "ocr_cache"
MIN_TEXT = 20     # a page with fewer extracted characters than this is treated as scanned
DPI = 110
PROMPT = ("Transcribe all text on this page exactly, in reading order. Put each heading on its "
          "own line and separate paragraphs with a blank line. Output only the text.")


def scanned_pages(path):
    with pymupdf.open(path) as pdf:
        return [i + 1 for i, p in enumerate(pdf) if len(p.get_text("text").strip()) < MIN_TEXT
                and (p.get_images() or p.get_drawings())]


def _client():
    import anthropic
    return anthropic.Anthropic(
        api_key=os.environ["OPENCODE_API_KEY"], timeout=120, max_retries=2,
        base_url=os.getenv("CLAUDE_BASE_URL", "https://opencode.ai/inference/anthropic"))


def ocr_pages(path, pages, workers=16):
    """Return {page_no: text} for the given 1-based pages."""
    if not pages:
        return {}
    digest = hashlib.sha1(Path(path).read_bytes()).hexdigest()[:16]
    model = os.getenv("OCR_MODEL", "claude-haiku-5-5")
    client = _client()

    def one(page_no):
        cached = CACHE_DIR / f"{digest}_{page_no}.txt"
        if cached.exists():
            return page_no, cached.read_text(encoding="utf-8")
        with pymupdf.open(path) as pdf:
            png = pdf[page_no - 1].get_pixmap(dpi=DPI).tobytes("png")
        with tracing.observation(f"ocr_page:{Path(path).name[:50]}:p{page_no}",
                                 as_type="generation", model=model,
                                 input={"page": page_no, "png_kb": len(png) // 1024}) as gen:
            r = client.messages.create(
                model=model, max_tokens=4000, thinking={"type": "disabled"},
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                 "data": base64.b64encode(png).decode()}},
                    {"type": "text", "text": PROMPT}]}])
            text = r.content[0].text if r.content else ""
            gen.update(output=tracing.clip(text, 2000),
                       usage_details={"input": r.usage.input_tokens,
                                      "output": r.usage.output_tokens})
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached.write_text(text, encoding="utf-8")
        return page_no, text

    out = {}
    with ThreadPoolExecutor(workers) as pool:
        for page_no, text in pool.map(tracing.bind(lambda p: _safe(one, p)), pages):
            if text:
                out[page_no] = text
    return out


def _safe(fn, page_no):
    try:
        return fn(page_no)
    except Exception:  # noqa: BLE001 - a failed page is just left without text
        return page_no, ""
