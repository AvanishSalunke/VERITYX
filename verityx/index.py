"""Build a named collection: PDFs -> chunks + Doc Cards on disk.

    python -m verityx.index Train/Train/Documents --collection train
    python -m verityx.index --collection train --cards-only
"""
import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .cards import build_cards, enrich_cards
from .chunker import chunk_pdf
from .embed import embed
from .ocr import ocr_pages, scanned_pages
from . import tracing

STORAGE = Path(__file__).resolve().parents[1] / "storage"


def collection_dir(name):
    return STORAGE / name


CONTEXTUAL = os.getenv("CONTEXTUAL_EMBED", "0") == "1"


def embed_text(chunk, card):
    """Section + text. CONTEXTUAL_EMBED=1 also prefixes document identity (title, issuer,
    period), the "contextual chunks" idea from metadata-driven RAG on FinanceBench. It
    measured slightly worse here (page_R 0.586 -> 0.568 on dev): the planner already picks the
    documents, and a per-document constant prefix only blurs ranking inside a document."""
    if not CONTEXTUAL:
        return f"{chunk['section']} {chunk['text']}"
    ident = " | ".join(x for x in [card.get("title", ""), card.get("issuer", ""),
                                   ", ".join(card.get("fiscal_periods", [])[:2])] if x)
    return f"{ident} | {chunk['section']} | {chunk['text']}"


def embed_chunks(ordered, cards):
    return embed(embed_text(c, cards[c["doc"]]) for c in ordered).astype(np.float16)


def build_collection(docs_dir, name):
    with tracing.trace("index", session_id=name, input={"docs_dir": str(docs_dir),
                                                        "collection": name}) as root:
        stats = _build(docs_dir, name)
        root.update(output=stats)
    tracing.flush()
    return stats


def _build(docs_dir, name):
    t0 = time.time()
    pdfs = sorted(Path(docs_dir).rglob("*.pdf"))
    with ProcessPoolExecutor() as pool:
        scanned = dict(zip(pdfs, pool.map(scanned_pages, pdfs)))
    with tracing.observation("ocr", input={p.name: len(v) for p, v in scanned.items() if v}):
        ocr = {p: ocr_pages(p, pages) for p, pages in scanned.items() if pages}
    t_ocr = time.time() - t0
    with ProcessPoolExecutor() as pool:
        per_doc = list(pool.map(chunk_pdf, pdfs, [ocr.get(p) for p in pdfs]))
    chunks_by_doc = defaultdict(list)
    for doc_chunks in per_doc:
        for c in doc_chunks:
            chunks_by_doc[c["doc"]].append(c)
    t_chunk = time.time() - t0

    cards = build_cards(chunks_by_doc)
    with tracing.observation("doc_cards", input={"docs": len(cards)}) as span:
        cards, enrich_status = enrich_cards(cards, chunks_by_doc)
        span.update(output=enrich_status)

    out = collection_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    ordered = [c for doc in sorted(chunks_by_doc) for c in chunks_by_doc[doc]]
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
        for n, c in enumerate(ordered):
            f.write(json.dumps({"id": n, **c}, ensure_ascii=False) + "\n")
    n = len(ordered)
    t_emb = time.time()
    np.save(out / "emb.npy", embed_chunks(ordered, cards))
    t_emb = time.time() - t_emb
    (out / "cards.json").write_text(json.dumps(cards, indent=1, ensure_ascii=False),
                                    encoding="utf-8")
    stats = {"docs": len(pdfs), "chunks": n, "chunk_s": round(t_chunk - t_ocr, 1),
             "ocr_pages": sum(len(v) for v in ocr.values()), "ocr_s": round(t_ocr, 1),
             "embed_s": round(t_emb, 1),
             "total_s": round(time.time() - t0, 1), "cards": enrich_status}
    (out / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    return stats


def rebuild_cards(name):
    """Rebuild cards.json (and the card-dependent embeddings) from saved chunks."""
    t0 = time.time()
    out = collection_dir(name)
    chunks_by_doc, ordered = defaultdict(list), []
    with open(out / "chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunks_by_doc[c["doc"]].append(c)
            ordered.append(c)
    cards, enrich_status = enrich_cards(build_cards(chunks_by_doc), chunks_by_doc)
    (out / "cards.json").write_text(json.dumps(cards, indent=1, ensure_ascii=False),
                                    encoding="utf-8")
    np.save(out / "emb.npy", embed_chunks(ordered, cards))
    return {"cards": enrich_status, "total_s": round(time.time() - t0, 1)}


def load_collection(name):
    d = collection_dir(name)
    with open(d / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    cards = json.loads((d / "cards.json").read_text(encoding="utf-8"))
    emb = np.load(d / "emb.npy").astype(np.float32)
    return chunks, cards, emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("docs_dir", nargs="?")
    ap.add_argument("--collection", required=True)
    ap.add_argument("--cards-only", action="store_true", help="rebuild Doc Cards from saved chunks")
    args = ap.parse_args()
    if args.cards_only:
        print(rebuild_cards(args.collection))
    else:
        print(build_collection(args.docs_dir, args.collection))


if __name__ == "__main__":
    main()
