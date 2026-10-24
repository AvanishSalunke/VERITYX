"""Query-time retrieval: route to documents via Doc Cards, then select chunks inside them.

route():  card match (query vs card text) fused with chunk evidence (best chunk scores per doc),
          plus a boost for docs whose periods match years named in the query.
select(): BM25 over chunks of the routed docs only, kept per doc by a relative score threshold
          so multi-doc questions get evidence from every routed doc.
"""
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .cards import STOP, card_text
from .embed import embed

TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
YEAR = re.compile(r"(?<!\d)(?:FY\s?)?((?:19|20)\d{2})(?!\d)", re.I)


def tokenize(text):
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


@dataclass
class Config:
    card_weight: float = 0.5      # share of card match vs chunk evidence in the routing score
    evidence_top: int = 5         # best chunks per doc summed as that doc's evidence
    year_boost: float = 0.3       # added when a doc's periods include a year named in the query
    doc_rel: float = 0.45         # keep docs scoring >= doc_rel * best doc
    max_docs: int = 6
    chunk_rel: float = 0.35       # keep chunks scoring >= chunk_rel * best chunk in that doc
    max_chunks_per_doc: int = 40
    max_chunks: int = 160
    lock_other_rel: float = 0.7   # when the query names a doc, others need >= this * best score
    dense_weight: float = 0.7     # share of embedding similarity vs BM25 in chunk scores
    plan_max_docs: int = 8        # cap on planner-chosen docs
    aspect_k: int = 12            # with a plan: chunks kept per (doc, aspect)
    section_hits: int = 2         # >0: a section with this many selected chunks pulls in its
    section_span: int = 1         #     other chunks within +-section_span pages of those hits
    max_expand: int = 40          #     keeping at most this many added chunks (best-scoring)


GENERIC = set("""annual report integrated fy pdf limited ltd india update statement summary
the and of for ver en re ar""".split())


def name_keys(card):
    """Identifying tokens for a doc: file name, title and issuer. (LLM entity lists also name
    auditors and banks, which would lock routing onto the wrong docs.)"""
    text = " ".join([card["doc"].rsplit(".", 1)[0], card.get("title", ""), card.get("issuer", "")])
    return {t for t in TOKEN.findall(text.lower().replace("_", " ").replace("-", " "))
            if len(t) >= 4 and not t.isdigit() and t not in GENERIC}


class Retriever:
    def __init__(self, chunks, cards, emb, config=None):
        self.cfg = config or Config()
        self.chunks = chunks
        self.emb = emb
        self.docs = sorted(cards)
        self.cards = cards
        # A name token that identifies at most 2 docs can lock routing onto them.
        keys = {d: name_keys(cards[d]) for d in self.docs}
        owners = defaultdict(set)
        for d, ks in keys.items():
            for k in ks:
                owners[k].add(d)
        self.lock_keys = {k: ds for k, ds in owners.items() if len(ds) <= 2}
        self.card_bm25 = BM25Okapi([tokenize(card_text(cards[d])) for d in self.docs])
        self.chunk_bm25 = BM25Okapi([tokenize(c["section"] + " " + c["text"]) for c in chunks])
        self.doc_of = np.array([self.docs.index(c["doc"]) for c in chunks])

    def _norm(self, x):
        m = x.max()
        return x / m if m > 0 else x

    def route(self, query, chunk_scores):
        cfg = self.cfg
        card = self._norm(self.card_bm25.get_scores(tokenize(query)))
        evidence = np.zeros(len(self.docs))
        for i in range(len(self.docs)):
            s = chunk_scores[self.doc_of == i]
            evidence[i] = np.sort(s)[-cfg.evidence_top:].sum() if len(s) else 0.0
        score = cfg.card_weight * card + (1 - cfg.card_weight) * self._norm(evidence)
        years = set(YEAR.findall(query))
        if years:
            for i, d in enumerate(self.docs):
                periods = " ".join(self.cards[d]["years"][:2] + self.cards[d]["fiscal_periods"])
                if any(y in periods or y[2:] in periods for y in years):
                    score[i] += cfg.year_boost
        order = np.argsort(-score)
        best = score[order[0]]
        q_tokens = set(TOKEN.findall(query.lower()))
        locked = {d for k in q_tokens & self.lock_keys.keys() for d in self.lock_keys[k]}
        if locked:
            others = [i for i in order if self.docs[i] not in locked
                      and score[i] >= cfg.lock_other_rel * best]
            picked = [self.docs.index(d) for d in sorted(locked)] + others
            picked = picked[:max(cfg.max_docs, len(locked))]
        else:
            picked = [i for i in order[:cfg.max_docs] if score[i] >= cfg.doc_rel * best]
        return ([self.docs[i] for i in picked],
                {self.docs[i]: round(float(score[i]), 3) for i in order[:10]}, sorted(locked))

    def chunk_scores(self, query):
        """Fused per-chunk score in [0, 1]: min-max BM25 blended with min-max cosine."""
        bm = self.chunk_bm25.get_scores(tokenize(query))
        cos = self.emb @ embed([query])[0]
        mm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)  # noqa: E731
        w = self.cfg.dense_weight
        return (1 - w) * mm(bm) + w * mm(cos)

    def search(self, query, plan=None):
        """Return (chunks, trace). With a planner `plan`, route to its docs and select chunks
        per aspect (aspect text + document-vocabulary keywords); otherwise fall back to
        deterministic routing and a single question-level score."""
        cfg = self.cfg
        base = self.chunk_scores(query)
        if plan:
            docs = [d for d in plan["docs"] if d in self.cards][:cfg.plan_max_docs]
            route_out = {"routed_docs": docs, "source": "planner", "periods": plan["periods"]}
            vectors = [("question", base)] + [
                (a["aspect"], self.chunk_scores(a["aspect"] + " " + " ".join(a["keywords"])))
                for a in plan["aspects"]]
            k = cfg.aspect_k
        else:
            docs, doc_scores, locked = self.route(query, base)
            route_out = {"routed_docs": docs, "source": "deterministic", "name_locked": locked,
                         "top_scores": doc_scores}
            vectors, k = [("question", base)], cfg.max_chunks_per_doc

        best = {}                       # chunk index -> best score across vectors
        why = defaultdict(set)          # chunk index -> vectors that selected it
        for d in docs:
            idx = np.where(self.doc_of == self.docs.index(d))[0]
            for name, vec in vectors:
                s = vec[idx]
                if not len(s) or s.max() <= 0:
                    continue
                keep = idx[s >= cfg.chunk_rel * s.max()]
                for i in keep[np.argsort(-vec[keep])][:k]:
                    best[int(i)] = max(best.get(int(i), 0.0), float(vec[i]))
                    why[int(i)].add(name)
        ranked = sorted(best, key=best.get, reverse=True)[:cfg.max_chunks]
        if cfg.section_hits:
            ranked += self._expand_sections(ranked, base)
        out = [self.chunks[i] for i in ranked]
        per_aspect = defaultdict(int)
        for i in ranked:
            for name in why[i]:
                per_aspect[name] += 1
        trace = [
            {"step_name": "doc_routing",
             "description": ("Planner-chosen docs from the Doc Card catalog" if plan else
                             "Score Doc Cards and per-doc chunk evidence; keep docs near the best"),
             "input": query, "output": route_out},
            {"step_name": "chunk_selection",
             "description": "BM25 + embedding scores within routed docs, top chunks per doc and "
                            "per aspect above a relative threshold",
             "input": {"docs": docs, "aspects": [n for n, _ in vectors]},
             "output": {"n_chunks": len(out), "per_doc": dict(_count(out)),
                        "per_aspect": dict(per_aspect)}},
        ]
        return out, trace


    def _expand_sections(self, ranked, scores):
        """Structure-aware recall: where several selected chunks share a section, add that
        section's other chunks on nearby pages (evidence clusters in notes/risk sections)."""
        cfg = self.cfg
        hits = defaultdict(list)
        for i in ranked:
            c = self.chunks[i]
            if c["section"]:
                hits[(c["doc"], c["section"])].append(c["page"])
        have = set(ranked)
        extra = []
        for (doc, sec), pages in hits.items():
            if len(pages) < cfg.section_hits:
                continue
            lo, hi = min(pages) - cfg.section_span, max(pages) + cfg.section_span
            for i in self._by_section.get((doc, sec), []):
                if i not in have and lo <= self.chunks[i]["page"] <= hi:
                    extra.append(i)
                    have.add(i)
        return sorted(extra, key=lambda i: -scores[i])[:cfg.max_expand]

    @property
    def _by_section(self):
        if not hasattr(self, "_section_index"):
            idx = defaultdict(list)
            for i, c in enumerate(self.chunks):
                idx[(c["doc"], c["section"])].append(i)
            self._section_index = idx
        return self._section_index


def _count(chunks):
    c = defaultdict(int)
    for ch in chunks:
        c[ch["doc"]] += 1
    return c
