"""Run the retrieval pipeline over the benchmark and write a run file for score.py.

    python eval/run_pipeline.py --collection train --name routed
    python eval/run_pipeline.py --collection train --name chunk_bm25 --no-route --k 88
    python eval/run_pipeline.py --collection train --name sweep --set doc_rel=0.5 chunk_rel=0.3
    python eval/run_pipeline.py --collection train --name planned --plan --workers 8
    python eval/run_pipeline.py --name ans --plan --answer --split dev20 --set aspect_k=12 max_chunks=160

With --answer the run file's chunks are the chunks the answer cites plus those the model
marked relevant; a sibling <name>_cited.jsonl holds the cited-only variant.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS, load_benchmark, load_splits  # noqa: E402
from verityx import tracing  # noqa: E402
from verityx.index import load_collection  # noqa: E402
from verityx.pipeline import run_query  # noqa: E402
from verityx.planner import Planner  # noqa: E402
from verityx.retrieve import Config, Retriever  # noqa: E402


def parse_overrides(pairs):
    types = {f.name: f.type for f in fields(Config)}
    out = {}
    for p in pairs:
        k, v = p.split("=", 1)
        out[k] = (int if types[k] in (int, "int") else float)(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="train")
    ap.add_argument("--name", required=True)
    ap.add_argument("--no-route", action="store_true", help="corpus-wide chunk BM25 baseline")
    ap.add_argument("--k", type=int, default=88, help="chunks for --no-route")
    ap.add_argument("--set", nargs="*", default=[], help="Config overrides, e.g. doc_rel=0.5")
    ap.add_argument("--plan", action="store_true", help="use the LLM query planner")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--answer", action="store_true", help="generate cited answers")
    ap.add_argument("--split", default="all", choices=["all", "dev", "dev20", "holdout"])
    args = ap.parse_args()

    t0 = time.time()
    chunks, cards, emb = load_collection(args.collection)
    cfg = Config(**parse_overrides(args.set))
    retriever = Retriever(chunks, cards, emb, cfg)
    planner = Planner(cards) if args.plan else None
    print(f"loaded {len(chunks)} chunks / {len(cards)} cards in {time.time() - t0:.1f}s; {cfg}")

    def fmt(cs):
        return [{"document": c["doc"], "pages": [c["page"]], "section": c["section"],
                 "text": c["text"]} for c in cs]

    def run_one(q):
        if args.no_route:
            t = time.time()
            s = retriever.chunk_scores(q["question"])
            picked = [chunks[i] for i in np.argsort(-s)[:args.k]]
            return {"id": q["id"], "answer": "", "chunks": fmt(picked), "relevant_extra": [],
                    "trace": [], "latency_s": round(time.time() - t, 3), "cost_usd": 0.0,
                    "tokens": 0}
        r = run_query(retriever, planner, q["question"], collection=args.collection,
                      do_answer=args.answer)
        return {"id": q["id"], "answer": r["answer"],
                "chunks": fmt(r["cited"] if args.answer else r["evidence"]),
                "relevant_extra": fmt(r["relevant"]), "trace": r["trace"],
                "latency_s": r["latency_s"], "cost_usd": r["cost_usd"], "tokens": r["tokens"],
                "langfuse_trace_url": r["langfuse_trace_url"]}

    with ThreadPoolExecutor(args.workers) as pool:
        bench = load_benchmark()
        if args.split != "all":
            keep = set(load_splits()[args.split])
            bench = [q for q in bench if q["id"] in keep]
        rows = list(pool.map(run_one, bench))
    failed = [r["id"] for r in rows if any(s["description"] == "failed" for s in r["trace"])]
    print(f"done in {time.time() - t0:.0f}s; LLM tokens {sum(r['tokens'] for r in rows)}"
          + (f"; FAILED steps for {failed}" if failed else ""))
    RUNS.mkdir(parents=True, exist_ok=True)

    def write(path, rows_out):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows_out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {path}")

    if args.answer:   # context = cited + relevant; cited-only kept for comparison
        write(RUNS / f"{args.name}.jsonl",
              [{**r, "chunks": r["chunks"] + r["relevant_extra"]} for r in rows])
        write(RUNS / f"{args.name}_cited.jsonl", rows)
    else:
        write(RUNS / f"{args.name}.jsonl", rows)
    tracing.flush()
    if tracing.enabled():
        print("langfuse: traced", len(rows), "queries, e.g.", rows[0].get("langfuse_trace_url"))


if __name__ == "__main__":
    main()
