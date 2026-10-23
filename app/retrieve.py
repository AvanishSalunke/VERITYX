"""Phase B step 2: contract-driven retrieval over a collection (dense + BM25, doc-restricted)."""
import json
import sqlite3
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from . import config, indexer
from .contract import EvidenceContract
from .tracing import observe

# every extra chunk costs context precision and answer tokens, so keep budgets tight
BUDGET = {"narrow": 4, "medium": 8, "broad": 20}  # chunks per requirement; tune on dev set
MAX_TOTAL = 30
RRF_K = 60


class Collection:
    def __init__(self, path: Path):
        self.con = sqlite3.connect(path / "meta.db", check_same_thread=False)
        self.index = faiss.read_index(str(path / "dense.faiss"))
        b = json.loads((path / "bm25_tokens.json").read_text())
        self.chunk_ids = b["chunk_ids"]
        self.vecs = self.index.reconstruct_n(0, self.index.ntotal)  # cached for masked search
        self.bm25 = BM25Okapi(b["tokens"])
        self.chunk_doc = dict(self.con.execute("select chunk_id, doc_id from chunks"))
        self.doc_of_pos = np.array([self.chunk_doc[c] for c in self.chunk_ids])
        self._docs = [{"doc_id": d, "filenames": json.loads(fn), "profile": json.loads(prof)}
                      for d, fn, prof in self.con.execute("select doc_id, filenames, profile from documents")]
        self.doc_name = {d["doc_id"]: d["filenames"][0] for d in self._docs}

    def docs(self) -> list[dict]:
        return self._docs

    def chunks(self, cids: list[int]) -> dict[int, dict]:
        if not cids:
            return {}
        rows = self.con.execute(
            f"select chunk_id,doc_id,pdf_page,kind,text from chunks where chunk_id in ({','.join('?' * len(cids))})",
            cids)
        return {cid: {"chunk_id": cid, "doc_id": d, "document": self.doc_name[d], "page": page,
                      "kind": kind, "text": text} for cid, d, page, kind, text in rows}


@lru_cache(maxsize=8)
def _load(path: str, mtime: float) -> Collection:
    return Collection(Path(path))


def load_collection(name: str) -> Collection:
    p = config.STORAGE_DIR / name
    return _load(str(p), (p / "meta.db").stat().st_mtime)


def search(col: Collection, text: str, doc_ids: list[str], n: int) -> list[tuple[int, float]]:
    """RRF fusion of dense and BM25 ranks, restricted to doc_ids (all docs if empty)."""
    mask = np.isin(col.doc_of_pos, doc_ids) if doc_ids else np.ones(len(col.chunk_ids), bool)
    pool = n * 4
    qv = indexer.embed([text])
    scores = col.vecs @ qv[0]
    scores = np.where(mask, scores, -1e9)
    dense = np.argsort(-scores)[:pool]
    bm = col.bm25.get_scores(indexer.tokenize(text))
    bm = np.where(mask, bm, -1e9)
    sparse = np.argsort(-bm)[:pool]
    fused: dict[int, float] = {}
    for ranks in (dense, sparse):
        for r, pos in enumerate(ranks):
            if mask[pos]:
                fused[int(pos)] = fused.get(int(pos), 0) + 1 / (RRF_K + r)
    top = sorted(fused.items(), key=lambda kv: -kv[1])[:n]
    return [(col.chunk_ids[p], s) for p, s in top]


@observe(name="execute_contract_retrieval")
def execute(col: Collection, contract: EvidenceContract) -> dict:
    selected: dict[int, dict] = {}
    per_req = {}
    for r in contract.requirements:
        docs = r.target_doc_ids or contract.target_doc_ids
        hits = search(col, r.search_text, docs, BUDGET[r.breadth])
        per_req[r.id] = [cid for cid, _ in hits]
        for cid, s in hits:
            selected.setdefault(cid, {"score": 0.0, "reqs": []})
            selected[cid]["score"] += s
            selected[cid]["reqs"].append(r.id)
    ranked = sorted(selected.items(), key=lambda kv: -kv[1]["score"])[:MAX_TOTAL]
    rows = col.chunks([cid for cid, _ in ranked])
    chunks = []
    for cid, meta in ranked:
        c = rows[cid]
        c["requirements"] = meta["reqs"]
        chunks.append(c)
    chunks.sort(key=lambda c: (c["document"], c["page"]))
    return {"chunks": chunks, "per_requirement": per_req}
