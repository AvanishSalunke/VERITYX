"""Phase A steps 2-3: document profiles and page gists (deterministic + LLM)."""
import json
import re
from concurrent.futures import ThreadPoolExecutor

from . import llm

YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
FY = re.compile(r"\b(?:FY|F\.Y\.?)\s?'?(\d{2,4}(?:\s?[-/–]\s?\d{2,4})?)", re.I)
SCOPE = re.compile(r"\b(consolidated|standalone)\b", re.I)
CURRENCY = re.compile(r"(₹|\$|€|£|\bRs\.?|\bINR\b|\bUSD\b|\bAED\b|\bSAR\b)")
UNITS = re.compile(r"\b(lakhs?|crores?|million|billion|thousand|basis points|bps|%)", re.I)


def deterministic_facts(text: str) -> dict:
    years = sorted(set(YEAR.findall(text)))
    return {
        "scope_counts": {w: len(re.findall(w, text, re.I)) for w in ("consolidated", "standalone")},
        "years": years[:20],
        "fiscal_periods": sorted({m.strip() for m in FY.findall(text)})[:10],
        "scope": sorted({m.lower() for m in SCOPE.findall(text)}),
        "currency": sorted(set(CURRENCY.findall(text)))[:5],
        "units": sorted({u.lower() for u in UNITS.findall(text)})[:8],
    }


PROFILE_SYS = ("You profile documents for a retrieval system. Use ONLY the given text. "
               "If a field is not stated, use null or []. Never guess. Respond only with JSON.")


def build_profile(doc_id: str, first_pages_text: str, det: dict) -> dict:
    user = (
        "From the opening pages below, return JSON with keys: entities (list of organisations/"
        "regulators/companies the document is about), reporting_periods (list), scope "
        "(Consolidated/Standalone/null), version_date (string or null), doc_type (e.g. annual "
        "report, regulatory proposal, accounting standard, research paper, FAQ, handbook), "
        "summary (max 2 sentences).\n\n"
        f"Deterministic hints: {json.dumps(det)}\n\nText:\n{first_pages_text[:9000]}"
    )
    p = llm.call_json(PROFILE_SYS, user, max_tokens=3000)
    # accept a scope only if the text uses that word often enough (avoid hallucinated scope)
    sc = (p.get("scope") or "")
    counts = det.get("scope_counts", {})
    if sc.lower() not in ("consolidated", "standalone") or counts.get(sc.lower(), 0) < 5:
        p["scope"] = None
    p["doc_id"] = doc_id
    p["deterministic"] = det
    return p


GIST_SYS = ("You index document pages. Use ONLY the page text. Do not infer or add facts. "
            "Respond only with JSON.")


def build_gist(page_text: str) -> dict:
    if len(page_text.strip()) < 40:
        return {"one_line_gist": "", "metrics_and_headers": [], "definitions_and_footnotes": []}
    user = (
        "Summarise the page below as JSON with exactly these keys:\n"
        '{"one_line_gist": "<=25 word summary of this page",\n'
        ' "metrics_and_headers": ["each metric name, table row/column header, unit, currency or period that literally appears"],\n'
        ' "definitions_and_footnotes": ["each defined term, policy, exception or footnote that literally appears"]}\n'
        "Fill the lists with actual text from the page (empty list if none). "
        "Never output the descriptions above.\n\nPage:\n" + page_text[:6000]
    )
    g = llm.call_json(GIST_SYS, user, max_tokens=3000)
    return {
        "one_line_gist": str(g.get("one_line_gist", "")),
        "metrics_and_headers": list(g.get("metrics_and_headers", []) or []),
        "definitions_and_footnotes": list(g.get("definitions_and_footnotes", []) or []),
    }


def build_gists(page_texts: list[str], workers: int = 8) -> list[dict]:
    def safe(t):
        try:
            return build_gist(t)
        except Exception as e:  # keep indexing alive; flag for review
            return {"one_line_gist": "", "metrics_and_headers": [],
                    "definitions_and_footnotes": [], "error": str(e)[:200]}
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(safe, page_texts))
