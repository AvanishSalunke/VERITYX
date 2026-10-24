"""Run the organizers' evaluator (unmodified) on a run file.

    python eval/official_eval.py eval/runs/ans_v1.jsonl --split dev20 --workers 5

The notebook's code cells are executed verbatim from Train/Train/Evaluation/Evaluation.ipynb.
Per the notebook's own notes only the configuration (API key) and the dataset are ours: the
example-usage cell is skipped, the key is taken from OPENCODE_API_KEY, and our run is passed
to the notebook's `evaluate_dataset`. To parallelize, each question is passed as its own
one-item dataset and the per-question metrics are averaged exactly as `evaluate_dataset` does.
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import verityx  # noqa: E402,F401 - loads .env
from common import DATA, EVAL, load_benchmark, load_splits  # noqa: E402

NOTEBOOK = DATA / "Evaluation" / "Evaluation.ipynb"
OUT_DIR = EVAL / "official"
OFFICIAL_CSV = EVAL / "official_results.csv"
METRICS = ["answer_precision", "answer_recall", "context_precision", "context_recall"]


def load_evaluator():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    ns = {}
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] != "code" or "sample_dataset" in src:
            continue
        exec(compile(src, str(NOTEBOOK), "exec"), ns)  # noqa: S102 - organizers' code, verbatim
    ns["OPENCODE_API_KEY"] = os.environ["OPENCODE_API_KEY"]      # the config cell is ours to set
    client = ns["OpenCodeClient"](api_key=ns["OPENCODE_API_KEY"])
    return ns["RAGMetricsEvaluator"](client), ns["evaluate_dataset"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--split", default="dev20")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    evaluator, evaluate_dataset = load_evaluator()
    bench = {q["id"]: q for q in load_benchmark()}
    preds = {}
    with open(args.run, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                preds[r["id"]] = r
    ids = [i for i in load_splits()[args.split] if i in preds]

    def item(qid):
        q, p = bench[qid], preds[qid]
        return {
            "query": q["question"],
            "ai_answer": p.get("answer", ""),
            "benchmark_answer": q["benchmark_answer"],
            "retrieved_chunks": [c["text"] for c in p.get("chunks", [])],
            "benchmark_chunks": [c["content"] for cs in q["benchmark_chunks_w_metadata"].values()
                                 for c in cs],
        }

    def run_one(qid):
        for attempt in range(2):
            try:
                res = evaluate_dataset(evaluator, [item(qid)], verbose=False)["per_question"][0]
                return qid, res, None
            except Exception as e:  # noqa: BLE001 - judge call failed; retry once
                err = str(e)
                time.sleep(2)
        return qid, None, err

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(run_one, ids))
    ok = [(qid, r) for qid, r, _ in results if r]
    failed = [(qid, e) for qid, r, e in results if not r]

    avg = {m: sum(r[m] for _, r in ok) / len(ok) for m in METRICS} if ok else {}
    print(f"\nOFFICIAL  {args.run.name}  split={args.split}  n={len(ok)}/{len(ids)}"
          f"  ({time.time() - t0:.0f}s)")
    for m in METRICS:
        print(f"  {m:18} {avg.get(m, float('nan')):.3f}")
    print("\n  per question       AnsP  AnsR  CtxP  CtxR   #chunks")
    for qid, r in sorted(ok):
        print(f"    {qid:5}          {r['answer_precision']:.2f}  {r['answer_recall']:.2f}  "
              f"{r['context_precision']:.2f}  {r['context_recall']:.2f}   "
              f"{len(preds[qid].get('chunks', []))}")
    for qid, e in failed:
        print(f"    {qid:5} FAILED: {e[:150]}")

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"{args.run.stem}_{args.split}.json"
    out.write_text(json.dumps({"summary": avg, "per_question": dict(ok),
                               "failed": dict(failed)}, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    new = not OFFICIAL_CSV.exists()
    with open(OFFICIAL_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "run", "split", "note", "n", *METRICS])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), args.run.name, args.split, args.note,
                    len(ok), *[round(avg[m], 4) if avg else None for m in METRICS]])
    print(f"\n  details -> {out}")


if __name__ == "__main__":
    main()
