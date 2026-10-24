"""Query planner: one LLM call that reads the question against the collection's Doc Card
catalog and returns which documents to search plus the aspects the answer must cover, each
with the vocabulary the documents themselves are likely to use.

The catalog goes first in the system prompt and is identical for every query of a collection,
so the provider can serve it from its prompt cache; only the question varies.
"""
from . import llm

PLAN_SYS = """You plan retrieval for a financial-research knowledge base. You see a catalog of
documents and a user question. Decide which documents contain the evidence, and break the
question into the aspects a complete answer must cover.

Rules:
- Choose documents by what they actually cover: entity, reporting period, jurisdiction,
  document type. Similar-looking documents from other years or entities are wrong choices.
- If the question spans a group ("the Indian companies in this set", "the US proposals"),
  include every catalog document in that group.
- Include supporting documents only if the question needs them (e.g. a regulation the
  question asks to apply). Prefer recall over precision only within the right group.
- Aspects: 3-8, each a distinct piece of information the answer needs. For each give
  "keywords": 4-10 terms or short phrases as they would appear IN THE DOCUMENT TEXT
  (statement line items, defined terms, section names), not the user's wording.
Respond only with JSON:
{"docs": [doc ids, most important first],
 "periods": ["periods the question targets, or empty"],
 "aspects": [{"aspect": "...", "keywords": ["..."]}]}

CATALOG
"""


def catalog_text(cards):
    """Stable numbered catalog; ids are positions in sorted doc order."""
    lines = []
    for i, doc in enumerate(sorted(cards)):
        c = cards[doc]
        lines.append(
            f"[{i}] {doc} | {c['title'][:120]} | type: {c['doc_type']} | issuer: "
            f"{c.get('issuer', '')[:80]} | period: {', '.join(c['fiscal_periods'][:3]) or '-'} | "
            f"{c['summary'][:260]} | covers: {'; '.join(c['answers'][:6])[:300]}")
    return "\n".join(lines)


class Planner:
    def __init__(self, cards):
        self.docs = sorted(cards)
        self.system = PLAN_SYS + catalog_text(cards)

    def _resolve(self, d):
        """Map a planner doc reference (catalog id, "[3]", file name, name sans .pdf) to a doc."""
        ref = str(d).strip().strip("[]").strip()
        if ref.isdigit():
            i = int(ref)
            return self.docs[i] if i < len(self.docs) else None
        low = ref.lower().removesuffix(".pdf")
        return next((doc for doc in self.docs if doc.lower().removesuffix(".pdf") == low), None)

    def plan(self, query):
        """Return (plan, meta). plan = {docs: [names], periods, aspects}; raises on failure."""
        out, meta = llm.call_json(self.system, f"Question:\n{query}", max_tokens=3000,
                                  reasoning_effort="none", name="query_planner")
        docs = []
        for d in out.get("docs") or []:
            name = self._resolve(d)
            if name and name not in docs:
                docs.append(name)
        aspects = [{"aspect": str(a.get("aspect", "")),
                    "keywords": [str(k) for k in a.get("keywords") or []]}
                   for a in out.get("aspects") or [] if isinstance(a, dict)]
        if not docs:
            raise ValueError("planner returned no valid docs")
        return {"docs": docs, "periods": out.get("periods") or [], "aspects": aspects}, meta
