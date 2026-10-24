# Eval harness (L0–L2, no LLM)

```bash
python eval/prep.py                       # once: extract 6,970 pages (~30s), ground gold, fix splits
python eval/baseline_bm25.py --k 40       # floor to beat
python eval/baseline_bm25.py --oracle     # upper bound / scorer sanity check
python eval/score.py eval/runs/<run>.jsonl --split dev --note "what changed"
```

Every scored run appends a row to `eval/results.csv`. Tune on `dev` (71 questions); score
`holdout` (29) only at milestones.

## Run file contract (one line per question)

```json
{"id": "A01", "answer": "...",
 "chunks": [{"document": "x.pdf", "pages": [3], "text": "..."}],
 "latency_s": 1.2, "cost_usd": 0.0004}
```

## Metrics

| Level | Metric | Meaning |
|---|---|---|
| L0 | `doc_P` / `doc_R` | Routed docs vs. gold docs |
| L1 | `ctx_P` | Share of our chunks on a gold (doc, page). Proxy for official context precision |
| L1 | `ctx_R` | Chunks on gold pages ÷ #gold chunks. Same formula as the official one, which is **not capped at 1** |
| L1 | `page_R` | Share of gold (doc, page) pairs covered |
| L2 | `num_R` / `num_P` | Numbers from the gold answer present in ours, and the reverse |

## Data quirks found during prep

- Gold `2026-05960.pdf` is `91fr14952.pdf` with **different pagination**. Its chunks are re-grounded by text match.
- Gold `fact-sheet-20260319.pdf` is `fact-sheet-SUMMARY.pdf`.
- Gold `Guidelines on Management of Problem Loans.pdf` is **not in the corpus** (21 chunks, unreachable).
- `B32` has no gold chunks.
- About 88 gold chunks per question (about 860 chars each, about 2 per page). Page-sized chunks cap `ctx_R` at about 0.58 even when retrieval is perfect.
