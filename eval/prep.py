"""One-time prep: extract PDF pages, ground every gold chunk to a real (doc, page), fix splits.

    python eval/prep.py            # skips page extraction if the cache exists
    python eval/prep.py --force    # re-extract pages

Grounding is needed because the benchmark's metadata does not always match the files on disk:
some gold docs carry a different file name (and pagination) than the PDF we were given, and
gold chunk text is markdown whose tables are laid out differently from raw PDF text. So each
chunk is matched to a page by word overlap rather than trusted blindly.
"""
import argparse
import gzip
import json
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import verityx  # noqa: E402,F401 - loads .env for the OCR client
from common import (ALIASES_FILE, CACHE, DOCS_DIR, GOLD_FILE, PAGES_FILE,  # noqa: E402
                    SPLITS_FILE, extract_numbers, load_benchmark, load_pages, normalize)

HOLDOUT_FRACTION = 0.3
SEED = 42
MATCH = 0.6        # fraction of a chunk's words that must appear on a page to call it that page
MIN_WORDS = 8      # chunks shorter than this are too small to ground reliably


def words(text):
    return {w for w in normalize(text).split() if len(w) > 3}


def _extract_one(path):
    # Scanned pages take their text from the pipeline's OCR (served from its cache after
    # indexing), so gold chunks on those pages can be grounded.
    from verityx.ocr import ocr_pages, scanned_pages
    ocr = ocr_pages(path, scanned_pages(path))
    with pymupdf.open(path) as pdf:
        return [{"doc": path.name, "page": i + 1,
                 "text": ocr.get(i + 1) or page.get_text("text")} for i, page in enumerate(pdf)]


def extract_pages():
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    n = 0
    with ProcessPoolExecutor() as pool, gzip.open(PAGES_FILE, "wt", encoding="utf-8") as out:
        for rows in pool.map(_extract_one, pdfs):
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    print(f"extracted {n} pages from {len(pdfs)} PDFs in {time.time() - t0:.1f}s")


class PageIndex:
    def __init__(self, pages):
        self.words = {(p["doc"], p["page"]): words(p["text"]) for p in pages}
        self.by_doc = defaultdict(list)
        for key in self.words:
            self.by_doc[key[0]].append(key)

    def overlap(self, gw, key):
        pw = self.words.get(key)
        return len(gw & pw) / len(gw) if pw else 0.0

    def best_in(self, gw, docs):
        return max(((self.overlap(gw, k), k) for d in docs for k in self.by_doc[d]),
                   default=(0.0, None))


def resolve_aliases(bench, index, sample=15):
    """Map each gold doc name not on disk to the on-disk PDF holding most of its chunks."""
    unknown = defaultdict(list)
    for row in bench:
        for c in (c for cs in row["benchmark_chunks_w_metadata"].values() for c in cs):
            if c["metadata"]["document"] not in index.by_doc:
                unknown[c["metadata"]["document"]].append(c)
    rng = random.Random(SEED)
    aliases = {}
    for name, chunks in unknown.items():
        votes = Counter()
        for c in rng.sample(chunks, min(sample, len(chunks))):
            gw = words(c["content"])
            if len(gw) >= MIN_WORDS:
                score, key = index.best_in(gw, index.by_doc)
                if score >= MATCH:
                    votes[key[0]] += 1
        aliases[name] = votes.most_common(1)[0][0] if votes else None
        print(f"alias: {name!r} ({len(chunks)} chunks) -> {aliases[name]!r} votes={dict(votes)}")
    ALIASES_FILE.write_text(json.dumps(aliases, indent=1), encoding="utf-8")
    return aliases


def ground_chunk(c, index, aliases):
    """Return ((doc, [pages]), status) for one gold chunk."""
    name = c["metadata"]["document"]
    listed = [int(p) for p in c["metadata"].get("page_numbers", [])]
    doc = name if name in index.by_doc else aliases.get(name)
    if doc is None:
        return (name, listed), "unreachable"       # PDF not in the corpus at all
    gw = words(c["content"])
    if len(gw) < MIN_WORDS:
        return (doc, listed), "too_short"          # trust metadata for tiny chunks
    if doc == name and max((index.overlap(gw, (doc, p)) for p in listed), default=0) >= MATCH:
        return (doc, listed), "exact"
    score, key = index.best_in(gw, [doc])
    if score >= MATCH:
        return (doc, [key[1]]), "regrounded"
    return (doc, listed), "unverified"


def build_gold(bench, index, aliases):
    gold, status = {}, Counter()
    for row in bench:
        pages, docs, unreachable, n_chunks = set(), set(), 0, 0
        for chunks in row["benchmark_chunks_w_metadata"].values():
            for c in chunks:
                n_chunks += 1
                (doc, ps), st = ground_chunk(c, index, aliases)
                status[st] += 1
                if st == "unreachable":
                    unreachable += 1
                    continue
                docs.add(doc)
                pages.update((doc, p) for p in ps)
        gold[row["id"]] = {
            "docs": sorted(docs),
            "pages": sorted(pages),
            "n_chunks": n_chunks,
            "n_unreachable": unreachable,
            "numbers": sorted(extract_numbers(row["benchmark_answer"])),
        }
    GOLD_FILE.write_text(json.dumps(gold, indent=1), encoding="utf-8")
    print(f"grounding of {sum(status.values())} gold chunks: {dict(status)}")
    return gold


def add_dev20(bench, n=20):
    """Fixed 20-question dev subset for the (slow, paid) official evaluator, stratified by
    question family (the A/B/C id prefix)."""
    splits = json.loads(SPLITS_FILE.read_text(encoding="utf-8"))
    if "dev20" in splits:
        return
    gold_chunks = {r["id"]: sum(len(c) for c in r["benchmark_chunks_w_metadata"].values())
                   for r in bench}
    fams = defaultdict(list)
    for qid in splits["dev"]:
        if gold_chunks[qid]:                     # skip questions with no gold chunks (B32)
            fams[qid[0]].append(qid)
    rng = random.Random(SEED)
    total = sum(len(v) for v in fams.values())
    picked = []
    for fam, ids in sorted(fams.items()):
        rng.shuffle(ids)
        picked += ids[:round(n * len(ids) / total)]
    splits["dev20"] = sorted(picked[:n])
    SPLITS_FILE.write_text(json.dumps(splits, indent=1), encoding="utf-8")
    print(f"dev20: {splits['dev20']}")


def make_splits(bench):
    if SPLITS_FILE.exists():
        print(f"splits exist, keeping {SPLITS_FILE.name}")
        add_dev20(bench)
        return
    # Stratify by number of gold docs so single- and multi-doc questions land in both splits.
    buckets = defaultdict(list)
    for row in bench:
        buckets[min(len(row["benchmark_chunks_w_metadata"]), 3)].append(row["id"])
    rng = random.Random(SEED)
    dev, holdout = [], []
    for ids in buckets.values():
        rng.shuffle(ids)
        k = round(len(ids) * HOLDOUT_FRACTION)
        holdout += ids[:k]
        dev += ids[k:]
    SPLITS_FILE.write_text(json.dumps({"dev": sorted(dev), "holdout": sorted(holdout)}, indent=1),
                           encoding="utf-8")
    print(f"splits: dev={len(dev)} holdout={len(holdout)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-extract PDF pages")
    args = ap.parse_args()

    bench = load_benchmark()
    if args.force or not PAGES_FILE.exists():
        extract_pages()
    t0 = time.time()
    index = PageIndex(load_pages())
    aliases = resolve_aliases(bench, index)
    gold = build_gold(bench, index, aliases)
    empty = [qid for qid, g in gold.items() if g["n_chunks"] == 0]
    print(f"gold: {len(gold)} questions, {sum(len(g['pages']) for g in gold.values())} (doc,page) "
          f"pairs, no-chunk questions: {empty} ({time.time() - t0:.0f}s)")
    make_splits(bench)


if __name__ == "__main__":
    main()
