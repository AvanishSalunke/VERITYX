"""Doc Cards: a compact profile per document used to route queries before chunk search.

The deterministic card (title, periods, scope, doc type, headings, distinctive terms) needs no
LLM and takes seconds for the whole corpus. If OPENCODE_API_KEY is set, `enrich_cards` adds a
short LLM summary, entities and the questions a document can answer (one call per document).
"""
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from . import llm, tracing

YEAR = re.compile(r"(?<!\d)(?:19[89]\d|20[0-4]\d)(?!\d)")
FY = re.compile(r"\b(?:FY|F\.Y\.?)\s?'?(\d{2,4}(?:\s?[-/–]\s?\d{2,4})?)", re.I)
SCOPE = re.compile(r"\b(consolidated|standalone)\b", re.I)
TOKEN = re.compile(r"[a-z][a-z0-9\-]{2,}")

# (doc_type, patterns matched against filename + opening pages); first match wins
DOC_TYPES = [
    ("annual report", r"annual report|integrated report|directors'? report|board'?s report"),
    ("regulatory proposal / rule", r"federal register|notice of proposed rulemaking|proposed rule"),
    ("accounting standard", r"accounting standards update|financial accounting standards board|asc \d"),
    ("supervisory handbook / guidance", r"handbook|interagency|frequently asked questions|\bfaq\b"),
    ("basel standard", r"basel committee|bcbs|basel framework"),
    ("board memo / fact sheet", r"\bmemo(randum)?\b|fact sheet"),
    ("research paper", r"\babstract\b|\bjournal\b|\barxiv\b|references"),
]
STOP = set("""the and for with that this from are was were has have had not but all any can
its may will which their such been also than other into more these those under shall each
per where when what there should would could only made upon them they our your about""".split())


def _years(text):
    return [y for y, _ in Counter(YEAR.findall(text)).most_common(6)]


def _doc_type(name, opening):
    hay = (name + " " + opening).lower()
    for label, pattern in DOC_TYPES:
        if re.search(pattern, hay):
            return label
    return "other"


def build_cards(chunks_by_doc, top_terms=40):
    """chunks_by_doc: {doc: [chunk, ...]} -> {doc: card}."""
    tf = {d: Counter(w for c in cs for w in TOKEN.findall(c["text"].lower()) if w not in STOP)
          for d, cs in chunks_by_doc.items()}
    df = Counter(w for counts in tf.values() for w in counts)
    n_docs = len(tf)
    cards = {}
    for doc, chunks in chunks_by_doc.items():
        full = " ".join(c["text"] for c in chunks)
        opening = " ".join(c["text"] for c in chunks if c["page"] <= 3)[:6000]
        headings = list(dict.fromkeys(c["section"] for c in chunks if c["section"]))
        step = max(1, len(headings) // 80)
        outline = headings[::step][:80]   # spread across the doc, not just the front matter
        total = sum(tf[doc].values()) or 1
        distinct = sorted(tf[doc], key=lambda w: -(tf[doc][w] / total) * math.log(n_docs / df[w]))
        scope = Counter(m.lower() for m in SCOPE.findall(full))
        cards[doc] = {
            "doc": doc,
            "title": headings[0] if headings else doc,
            "doc_type": _doc_type(doc, opening),
            "years": _years(doc + " " + opening) or _years(full),
            "fiscal_periods": sorted({m.strip() for m in FY.findall(doc + " " + opening)})[:6],
            "scope": [s for s, n in scope.items() if n >= 5],
            "n_pages": max((c["page"] for c in chunks), default=0),
            "headings": outline,
            "terms": distinct[:top_terms],
            "issuer": "",
            "summary": "",
            "entities": [],
            "answers": [],
        }
    return cards


def card_text(card):
    """Text used to match a query against a card."""
    parts = [card["doc"].replace("_", " ").replace("-", " "), card["title"], card["doc_type"],
             " ".join(card["years"]), " ".join(card["fiscal_periods"]), " ".join(card["scope"]),
             card.get("issuer", ""), card["summary"], " ".join(card["entities"]), " ".join(card["answers"]),
             " ".join(card["headings"]), " ".join(card["terms"])]
    return " ".join(p for p in parts if p)


# ---------- optional LLM enrichment ----------

CARD_SYS = ("You profile documents for a retrieval system. Use ONLY the given text; if a field "
            "is not stated use null or []. Never guess. Respond only with JSON.")


def _enrich_one(card, opening):
    user = (
        "Return JSON with keys:\n"
        "  title: the document's real title\n"
        "  issuer: who published it (company, regulator, authors)\n"
        "  entities: organisations the document is about\n"
        "  reporting_period: period covered, e.g. 'FY2025-26' or 'proposal dated March 2026', "
        "or null\n"
        "  summary: max 2 sentences on what it is and what it covers\n"
        "  answers: 8-12 short topics a reader could answer from this document\n\n"
        f"File name: {card['doc']}\nDocument type guess: {card['doc_type']}\n"
        f"Outline (headings sampled across the document): {'; '.join(card['headings'][:80])}\n\n"
        f"Opening text:\n{opening[:7000]}")
    out, meta = llm.call_json(CARD_SYS, user, max_tokens=2500,
                              name=f"doc_card_enrichment:{card['doc'][:60]}")
    if out.get("title"):
        card["title"] = out["title"]
    card["issuer"] = out.get("issuer") or ""
    card["summary"] = out.get("summary") or ""
    card["entities"] = [e for e in out.get("entities") or [] if isinstance(e, str)]
    card["answers"] = [a for a in out.get("answers") or [] if isinstance(a, str)]
    if out.get("reporting_period"):
        card["fiscal_periods"] = sorted(set(card["fiscal_periods"]) | {out["reporting_period"]})
    return card, meta


def enrich_cards(cards, chunks_by_doc, workers=16):
    """Add LLM title/issuer/summary/entities/answers to each card; no-op without a key."""
    if not llm.available():
        return cards, "skipped (no OPENCODE_API_KEY)"

    def job(doc):
        opening = " ".join(c["text"] for c in chunks_by_doc[doc] if c["page"] <= 4)
        try:
            return _enrich_one(cards[doc], opening)[1], None
        except Exception as e:  # noqa: BLE001 - a failed card keeps its deterministic fields
            return None, f"{doc}: {e}"

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(tracing.bind(job), list(cards)))
    errors = [err for _, err in results if err]
    metas = [m for m, _ in results if m]
    tokens = sum(m["usage"].get("total_tokens", 0) for m in metas if not m["cached"])
    return cards, (f"enriched {len(metas)}/{len(cards)}, {tokens} tokens, "
                   f"${sum(m['cost_usd'] for m in metas):.4f}"
                   + (f", errors: {errors[:3]}" if errors else ""))
