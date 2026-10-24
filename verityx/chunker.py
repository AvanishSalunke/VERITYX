"""PDF -> heading-aware, page-bound chunks.

Each chunk is the text between two headings on one page (chunks never cross pages, matching
the benchmark's chunk granularity), with the most recent heading carried as its section.
Headings are found from font size / boldness relative to the document's body text; running
headers and footers are detected by repetition across pages and dropped.
"""
import re
from collections import Counter

import pymupdf

MIN_CHARS = 150     # segments shorter than this merge into the next one on the same page
MAX_CHARS = 1800    # longer segments split at paragraph (block) boundaries
REPEAT_SHARE = 0.3  # a line on >30% of pages is a running header/footer

_SPACE = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")


def _lines(page):
    """Yield (block_no, text, size, bold) for each text line on the page."""
    for b_no, block in enumerate(page.get_text("dict")["blocks"]):
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = _SPACE.sub(" ", "".join(s["text"] for s in spans)).strip()
            size = max(s["size"] for s in spans)
            bold = all(s["flags"] & 16 or "bold" in s["font"].lower() for s in spans)
            yield b_no, text, round(size, 1), bold


def _signature(text):
    """Line identity for header/footer detection; page numbers vary, so ignore digits."""
    return _DIGITS.sub("#", text.lower())


def _is_heading(text, size, bold, body):
    if len(text) > 120 or len(text) < 3 or not any(c.isalpha() for c in text):
        return False
    if text.endswith((",", ";")) or text[0].islower():
        return False
    if size >= body + 1.5:
        return True
    return bold and size >= body - 0.1 and len(text) <= 90 and not text.endswith(".")


def _ocr_lines(text, body):
    """OCR text -> the (block_no, text, size, bold) lines the layout path produces: one block
    per paragraph, and short title-like lines given a heading-sized font."""
    out = []
    for b_no, para in enumerate(p for p in re.split(r"\n\s*\n", text) if p.strip()):
        for line in para.splitlines():
            line = _SPACE.sub(" ", line).strip(" #*")
            if not line:
                continue
            heading = (len(line) <= 90 and not line.endswith((".", ",", ";", ":"))
                       and (line[0].isupper() or line[0].isdigit()) and len(line.split()) <= 12)
            out.append((b_no, line, body + 2 if heading else body, False))
    return out


def chunk_pdf(path, ocr_text=None):
    """Return a list of chunks: {doc, page, section, text}. `ocr_text` ({page_no: text})
    supplies text for scanned pages that have no text layer."""
    doc_name = path.name
    ocr_text = ocr_text or {}
    with pymupdf.open(path) as pdf:
        pages = [list(_lines(p)) for p in pdf]

    sizes = Counter()
    sig_pages = Counter()
    for lines in pages:
        for _, text, size, _ in lines:
            sizes[size] += len(text)
        sig_pages.update({_signature(t) for _, t, _, _ in lines})
    body = sizes.most_common(1)[0][0] if sizes else 10.0
    for page_no, text in ocr_text.items():
        if not pages[page_no - 1]:
            pages[page_no - 1] = _ocr_lines(text, body)
    repeated = ({s for s, n in sig_pages.items() if n > REPEAT_SHARE * len(pages)}
                if len(pages) >= 5 else set())

    chunks, section = [], ""
    for page_no, lines in enumerate(pages, start=1):
        segments = []          # [section, [paragraph texts]]
        current = [section, []]
        para, last_block = [], None
        for b_no, text, size, bold in lines:
            if _signature(text) in repeated:
                continue
            if _is_heading(text, size, bold, body):
                if para:
                    current[1].append(" ".join(para))
                    para = []
                if current[1]:
                    segments.append(current)
                section = text
                current = [section, []]
                last_block = b_no
                continue
            if b_no != last_block and para:
                current[1].append(" ".join(para))
                para = []
            para.append(text)
            last_block = b_no
        if para:
            current[1].append(" ".join(para))
        if current[1]:
            segments.append(current)
        chunks.extend(_finalize(doc_name, page_no, segments))
    return chunks


def _finalize(doc_name, page_no, segments):
    """Merge tiny segments forward, split huge ones at paragraph boundaries."""
    out, carry = [], ""
    for sec, paras in segments:
        paras = ([carry] if carry else []) + paras
        text = " ".join(paras)
        if len(text) < MIN_CHARS:
            carry = text
            continue
        carry = ""
        buf = ""
        for p in ([text] if len(text) <= MAX_CHARS else paras):
            if buf and len(buf) + len(p) > MAX_CHARS:
                out.append({"doc": doc_name, "page": page_no, "section": sec, "text": buf})
                buf = ""
            buf = (buf + " " + p).strip()
        if buf:
            out.append({"doc": doc_name, "page": page_no, "section": sec, "text": buf})
    if carry:
        if out:
            out[-1]["text"] += " " + carry
        else:
            out.append({"doc": doc_name, "page": page_no,
                        "section": segments[-1][0] if segments else "", "text": carry})
    return out
