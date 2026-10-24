# VERITYX

A knowledge base for agents over a messy dump of financial documents. It picks the documents
a question needs before searching them, and answers only from cited evidence.

## How a query works

```
INDEX   PDFs -> page-bound, heading-aware chunks (vision-LLM OCR for scanned pages)
             -> Doc Card per PDF: title, issuer, period, scope, topics (1 LLM call each)
             -> BM25 + model2vec embeddings over chunks
QUERY   1. planner     one LLM call over the Doc Card catalog: target docs, periods, aspects
        2. retrieval   BM25 + embeddings inside those docs only, top chunks per aspect,
                       plus section expansion (more chunks from sections that already match)
        3. answer      one LLM call, every claim cites [chunk ids]; cited + relevant chunks
                       are returned as context
        4. verifier    drops claims with no valid citation
```

Each step goes into the response `trace` array, and the whole query is one Langfuse trace.

| Module | Role |
|---|---|
| `verityx/chunker.py`, `ocr.py` | PDF -> chunks; OCR fallback for pages with no text layer |
| `verityx/cards.py` | Doc Cards (deterministic + LLM enrichment) |
| `verityx/index.py` | Builds a named collection under `storage/<name>/` |
| `verityx/planner.py`, `retrieve.py` | Document routing and chunk selection |
| `verityx/answer.py` | Cited answer generation and citation verifier |
| `verityx/pipeline.py` | `run_query()`: one query end to end (the `/query` response shape) |
| `verityx/api.py` | FastAPI app: `POST /index`, `POST /query` |
| `verityx/llm.py`, `tracing.py` | OpenCode client (disk cache) and Langfuse telemetry |

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` in the repo root:

```
OPENCODE_API_KEY=...
OPENCODE_BASE_URL=https://opencode.ai/inference/openai/v1/chat/completions
OPENCODE_MODEL=deepseek-v4.1-flash
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
# optional
LLM_PRICE_IN=      # USD per 1M input tokens, for cost tracking
LLM_PRICE_OUT=     # USD per 1M output tokens
LLM_CACHE=1        # 0 = always call the LLM (use for cost/latency exports)
```

## API

```bash
uvicorn verityx.api:app --host 0.0.0.0 --port 8000
```

The app exposes exactly two endpoints:

```bash
curl -X POST http://HOST:8000/index -F "file=@docs.zip" -F "collection_name=test_corpus"
curl -X POST http://HOST:8000/query -H "Content-Type: application/json" \
     -d '{"collection_name": "test_corpus", "query": "..."}'
```

- **`/index`** returns `status: success` once indexing has finished. Each collection is stored in
  its own `storage/<collection_name>/`, so indexing one collection never touches another.
- **`/query`** returns:
  - `answer`
  - `chunks`: each with `document`, `page`, `section`, `text` and `cited_in_answer`
  - `trace`: the steps, each with exactly `step_name`, `description`, `input` and `output`
  - `latency_s`, `cost_usd`, `tokens` and `langfuse_trace_url`

**Ingestion time:** the training corpus (35 PDFs, 6,970 pages, 105 of them OCR'd) indexes in
about 4.5 min. Small corpora take seconds (3 PDFs: 9 s). Estimate: about 10 min for a corpus of
similar size, which allows for OCR-heavy and slower servers.

## Run

```bash
python -m verityx.index Train/Train/Documents --collection train   # ~4.5 min incl. OCR
python -m verityx.index --collection train --cards-only            # rebuild Doc Cards only
python eval/run_pipeline.py --name ans --plan --answer --split dev20 --workers 4
```

## Evaluate

See [eval/README.md](eval/README.md) for the run-file format and metrics.

```bash
python eval/prep.py                                       # once: ground gold chunks to pages
python eval/score.py eval/runs/ans.jsonl --split dev20    # free L0-L2 proxies (seconds)
python eval/official_eval.py eval/runs/ans.jsonl --split dev20   # organizers' evaluator (~15 min)
```

`official_eval.py` runs the evaluator notebook's code cells unmodified; only the API key and
the dataset are supplied. The splits are fixed in `eval/splits.json`: dev (71), holdout (29) and
dev20 (a stratified dev subset). Results are logged to `eval/results.csv` and
`eval/official_results.csv`.

## Current results (official evaluator, dev20)

| Answer P | Answer R | Context P | Context R |
|---|---|---|---|
| 0.36 | 0.32 | 0.73 | 0.47 |

Each query takes about 8–11 s and about 35–50k tokens: planner ~6k, answer ~30–45k.
