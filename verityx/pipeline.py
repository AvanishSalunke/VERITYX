"""One query end to end: plan -> retrieve -> answer -> verify, traced to Langfuse.

Returns the /query response shape: answer, chunks used (with document and page), and the
step-by-step `trace` array (step_name / description / input / output). The same steps are
recorded as nested Langfuse observations under one trace per query.
"""
import time

from . import answer as answerer
from . import tracing


def run_query(retriever, planner, query, collection="", do_answer=True):
    t0 = time.time()
    steps, cost, tokens = [], 0.0, 0
    plan, text, cited, relevant = None, "", [], []
    with tracing.trace("query", session_id=collection or None, input={"query": query},
                       metadata={"collection": collection}) as root:
        if planner:
            try:
                plan, meta = planner.plan(query)
                cost += meta["cost_usd"]
                tokens += meta["usage"].get("total_tokens", 0)
                steps.append({"step_name": "query_planner",
                              "description": "LLM reads the question against the Doc Card "
                                             "catalog: target docs, periods, aspects",
                              "input": query,
                              "output": {**plan, "tokens": meta["usage"].get("total_tokens", 0),
                                         "latency_s": meta["latency_s"],
                                         "cached": meta["cached"]}})
            except Exception as e:  # noqa: BLE001 - fall back to deterministic routing
                steps.append({"step_name": "query_planner", "description": "failed",
                              "input": query, "output": str(e)})

        with tracing.observation("retrieval", as_type="retriever",
                                 input={"query": query, "plan_docs": (plan or {}).get("docs")}
                                 ) as span:
            picked, retrieval_steps = retriever.search(query, plan)
            span.update(output={s["step_name"]: s["output"] for s in retrieval_steps})
        steps += retrieval_steps

        if do_answer:
            try:
                text, cited, relevant, meta = answerer.generate(query, picked, plan)
                cost += meta["cost_usd"]
                tokens += meta["usage"].get("total_tokens", 0)
                steps.append({
                    "step_name": "llm_answer_generation",
                    "description": "Answer from the wide evidence set with [chunk] citations; "
                                   "cited + model-marked relevant chunks become the context",
                    "input": {"n_evidence": len(picked),
                              "aspects": [a["aspect"] for a in (plan or {}).get("aspects", [])]},
                    "output": {"n_cited": meta["n_cited"],
                               "n_relevant_uncited": meta["n_relevant_uncited"],
                               "tokens": meta["usage"].get("total_tokens", 0),
                               "latency_s": meta["latency_s"], "cached": meta["cached"]}})
                verifier = {"n_dropped": meta["n_dropped_unsupported"],
                            "dropped": meta["dropped_unsupported"][:20]}
                with tracing.observation("citation_verifier", input="answer draft",
                                         output=verifier):
                    pass
                steps.append({"step_name": "citation_verifier",
                              "description": "Drop answer claims that carry no valid chunk "
                                             "citation", "input": "answer draft",
                              "output": verifier})
            except Exception as e:  # noqa: BLE001 - recorded; the query returns no answer
                steps.append({"step_name": "llm_answer_generation", "description": "failed",
                              "input": len(picked), "output": str(e)})

        used = cited + relevant if do_answer else picked
        latency = round(time.time() - t0, 3)
        root.update(output={"answer": tracing.clip(text, 4000), "n_chunks": len(used)},
                    metadata={"latency_s": latency, "cost_usd": cost, "tokens": tokens})
        trace_url = tracing.current_trace_url()

    return {"answer": text, "chunks": used, "cited": cited, "relevant": relevant,
            "evidence": picked, "trace": steps, "latency_s": latency, "cost_usd": cost,
            "tokens": tokens, "langfuse_trace_url": trace_url}
