"""Reference runs that exercise the harness (no LLM, so they have no answer text).

    python eval/baseline_bm25.py --k 40          # corpus-wide page-level BM25 -> runs/bm25_k40.jsonl
    python eval/baseline_bm25.py --oracle        # gold pages as chunks -> runs/oracle.jsonl (upper bound)

The BM25 run is the "plain retrieval" floor our pipeline must beat; the oracle run checks
that the scorer gives ~1.0 when retrieval is perfect.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS, load_benchmark, load_gold, load_pages  # noqa: E402

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN.findall(text.lower())


def write_run(path, rows):
    RUNS.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {path}")


def bm25_run(k):
    pages = load_pages()
    t0 = time.time()
    bm25 = BM25Okapi([tokenize(p["text"]) for p in pages])
    print(f"indexed {len(pages)} pages in {time.time() - t0:.1f}s")
    rows = []
    for q in load_benchmark():
        t = time.time()
        scores = bm25.get_scores(tokenize(q["question"]))
        top = sorted(range(len(pages)), key=scores.__getitem__, reverse=True)[:k]
        rows.append({
            "id": q["id"],
            "answer": "",
            "chunks": [{"document": pages[i]["doc"], "pages": [pages[i]["page"]],
                        "text": pages[i]["text"]} for i in top],
            "latency_s": round(time.time() - t, 3),
            "cost_usd": 0.0,
        })
    write_run(RUNS / f"bm25_k{k}.jsonl", rows)


def oracle_run():
    answers = {q["id"]: q["benchmark_answer"] for q in load_benchmark()}
    rows = [{"id": qid, "answer": answers[qid],
             "chunks": [{"document": d, "pages": [p], "text": ""} for d, p in g["pages"]]}
            for qid, g in load_gold().items()]
    write_run(RUNS / "oracle.jsonl", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--oracle", action="store_true")
    args = ap.parse_args()
    oracle_run() if args.oracle else bm25_run(args.k)


if __name__ == "__main__":
    main()
