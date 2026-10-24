"""Score a run file with the free (no-LLM) proxies L0-L2.

    python eval/score.py eval/runs/bm25.jsonl --split dev --note "bm25 k=40"

Run file contract: one JSON object per line
    {"id": "A01",
     "answer": "...",                                   # final answer text
     "chunks": [{"document": "x.pdf", "pages": [3, 4], "text": "..."}],
     "latency_s": 1.2, "cost_usd": 0.0004}              # optional

Metrics (averaged over the split):
    L0  doc_P / doc_R      routed docs vs gold docs
    L1  ctx_P              share of our chunks that touch a gold (doc, page)       ~ official context precision
        ctx_R              chunks touching gold / #gold chunks (same formula as the official
                           evaluator, which is NOT capped at 1); ctx_R_cap is min(1, ctx_R)
        page_R             share of gold (doc, page) pairs our chunks cover
    L2  num_R / num_P      numbers in the gold answer found in ours, and ours found in gold
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_CSV, extract_numbers, load_gold, load_splits  # noqa: E402

METRICS = ["doc_P", "doc_R", "ctx_P", "ctx_R", "ctx_R_cap", "page_R", "num_R", "num_P"]


def ratio(num, den):
    return num / den if den else None


def score_question(gold, pred):
    gold_docs, gold_pages = set(gold["docs"]), {tuple(p) for p in gold["pages"]}
    chunks = pred.get("chunks", [])
    our_docs = {c["document"] for c in chunks}
    our_pages = {(c["document"], int(p)) for c in chunks for p in c.get("pages", [])}
    hits = sum(any((c["document"], int(p)) in gold_pages for p in c.get("pages", [])) for c in chunks)
    gold_nums, our_nums = set(gold["numbers"]), extract_numbers(pred.get("answer", ""))
    ctx_r = ratio(hits, gold["n_chunks"])
    return {
        "doc_P": ratio(len(our_docs & gold_docs), len(our_docs)) if chunks else 0.0,
        "doc_R": ratio(len(our_docs & gold_docs), len(gold_docs)),
        "ctx_P": ratio(hits, len(chunks)) if chunks else 0.0,
        "ctx_R": ctx_r,
        "ctx_R_cap": None if ctx_r is None else min(1.0, ctx_r),
        "page_R": ratio(len(our_pages & gold_pages), len(gold_pages)),
        "num_R": ratio(len(our_nums & gold_nums), len(gold_nums)),
        "num_P": ratio(len(our_nums & gold_nums), len(our_nums)) if pred.get("answer") else None,
        "missed_docs": sorted(gold_docs - our_docs),
        "n_chunks": len(chunks),
    }


def diagnose(s):
    if s["page_R"] is None:
        return "no gold chunks for this question"
    if s["missed_docs"]:
        return "missed docs: " + ", ".join(s["missed_docs"])
    if (s["page_R"] or 0) < 0.5:
        return "right docs, missed most gold pages"
    if (s["ctx_P"] or 0) < 0.5:
        return "noisy: most chunks off gold pages"
    return "answer gap" if (s["num_R"] or 1) < 0.5 else "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--split", default="dev", choices=["dev", "dev20", "holdout", "all"])
    ap.add_argument("--note", default="")
    ap.add_argument("--worst", type=int, default=10)
    args = ap.parse_args()

    gold = load_gold()
    ids = sorted(gold) if args.split == "all" else load_splits()[args.split]
    preds = {}
    with open(args.run, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                preds[row["id"]] = row

    per_q = {qid: score_question(gold[qid], preds.get(qid, {})) for qid in ids}
    missing = [qid for qid in ids if qid not in preds]

    summary = {}
    for m in METRICS:
        vals = [s[m] for s in per_q.values() if s[m] is not None]
        summary[m] = sum(vals) / len(vals) if vals else None
    lat = [preds[q]["latency_s"] for q in ids if "latency_s" in preds.get(q, {})]
    cost = [preds[q]["cost_usd"] for q in ids if "cost_usd" in preds.get(q, {})]
    summary["avg_chunks"] = sum(s["n_chunks"] for s in per_q.values()) / len(per_q)
    summary["avg_latency_s"] = sum(lat) / len(lat) if lat else None
    summary["avg_cost_usd"] = sum(cost) / len(cost) if cost else None

    fmt = lambda v: "  -  " if v is None else f"{v:.3f}"  # noqa: E731
    print(f"\n{args.run.name}  split={args.split}  n={len(ids)}"
          + (f"  MISSING={len(missing)}" if missing else ""))
    print("  L0 routing   doc_P " + fmt(summary["doc_P"]) + "   doc_R " + fmt(summary["doc_R"]))
    print("  L1 context   ctx_P " + fmt(summary["ctx_P"]) + "   ctx_R " + fmt(summary["ctx_R"])
          + "  (capped " + fmt(summary["ctx_R_cap"]) + ")   page_R " + fmt(summary["page_R"]))
    print("  L2 answer    num_R " + fmt(summary["num_R"]) + "   num_P " + fmt(summary["num_P"]))
    print(f"  avg chunks {summary['avg_chunks']:.1f}   latency {fmt(summary['avg_latency_s'])}s"
          f"   cost ${fmt(summary['avg_cost_usd'])}")

    worst = sorted(per_q.items(), key=lambda kv: ((kv[1]["page_R"] or 0) + (kv[1]["ctx_P"] or 0)))
    print(f"\n  worst {args.worst} (by page_R + ctx_P):")
    for qid, s in worst[:args.worst]:
        print(f"    {qid:5} page_R {fmt(s['page_R'])} ctx_P {fmt(s['ctx_P'])} doc_R {fmt(s['doc_R'])}"
              f"  -> {diagnose(s)}")

    new = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "run", "split", "note", *METRICS, "avg_chunks", "avg_latency_s",
                        "avg_cost_usd"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), args.run.name, args.split, args.note,
                    *[None if summary[m] is None else round(summary[m], 4) for m in METRICS],
                    round(summary["avg_chunks"], 1), summary["avg_latency_s"], summary["avg_cost_usd"]])


if __name__ == "__main__":
    main()
