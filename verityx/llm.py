"""OpenCode chat client returning parsed JSON, with a disk cache and token accounting.

Identical (model, system, user, max_tokens) requests are served from storage/llm_cache, so
re-running the benchmark after a non-LLM change costs nothing. Every call reports its usage;
cost is computed from LLM_PRICE_IN / LLM_PRICE_OUT (USD per 1M tokens) when those are set.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import requests

from . import tracing

CACHE_DIR = Path(__file__).resolve().parents[1] / "storage" / "llm_cache"


def _cfg():
    return {
        "key": os.getenv("OPENCODE_API_KEY", ""),
        "url": os.getenv("OPENCODE_BASE_URL",
                         "https://opencode.ai/inference/openai/v1/chat/completions"),
        "model": os.getenv("OPENCODE_MODEL", "deepseek-v4.1-flash"),
        "price_in": float(os.getenv("LLM_PRICE_IN", "0") or 0),
        "price_out": float(os.getenv("LLM_PRICE_OUT", "0") or 0),
    }


def available():
    return bool(os.getenv("OPENCODE_API_KEY"))


def cost_of(usage):
    c = _cfg()
    return (usage.get("prompt_tokens", 0) * c["price_in"]
            + usage.get("completion_tokens", 0) * c["price_out"]) / 1e6


def _parse(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json")
    return json.loads(text)


def call_json(system, user, max_tokens=2000, retries=3, timeout=90, use_cache=True,
              reasoning_effort=None, name="llm_call"):
    """Return (parsed_json, meta) where meta has usage, latency_s, cost_usd, cached.

    reasoning_effort ("none", "low", ...) is passed through to the model; on the query path
    "none" matters: with default reasoning the planner took ~28s median and sometimes spent
    its whole token budget thinking, versus ~3s without.

    Each call is a Langfuse `generation` named `name` (model, usage, cost); a cache hit is
    logged with zero usage so traced spend matches what was actually billed."""
    model = _cfg()["model"]
    params = {"max_tokens": max_tokens, "temperature": 0,
              "reasoning_effort": reasoning_effort or "default"}
    with tracing.observation(name, as_type="generation", model=model, model_parameters=params,
                             input=[{"role": "system", "content": tracing.clip(system, 3000)},
                                    {"role": "user", "content": tracing.clip(user)}]) as gen:
        try:
            out, meta = _call(system, user, max_tokens, retries, timeout, use_cache,
                              reasoning_effort)
        except Exception as e:
            gen.update(level="ERROR", status_message=str(e)[:500])
            raise
        usage = {} if meta["cached"] else meta["usage"]
        # Send cost only when LLM_PRICE_* is configured; otherwise let Langfuse compute it from
        # a model price defined in its UI (an explicit 0 would override that).
        priced = bool(_cfg()["price_in"] or _cfg()["price_out"])
        gen.update(output=out, usage_details=tracing.usage_details(usage),
                   metadata={"cached": meta["cached"], "latency_s": meta["latency_s"]},
                   **({"cost_details": {"total": 0.0 if meta["cached"] else meta["cost_usd"]}}
                      if priced else {}))
        return out, meta


def _call(system, user, max_tokens, retries, timeout, use_cache, reasoning_effort):
    c = _cfg()
    if not c["key"]:
        raise RuntimeError("OPENCODE_API_KEY is not set")
    key = hashlib.sha256(json.dumps([c["model"], system, user, max_tokens, reasoning_effort])
                         .encode()).hexdigest()
    path = CACHE_DIR / f"{key}.json"
    use_cache = use_cache and os.getenv("LLM_CACHE", "1") != "0"   # LLM_CACHE=0: real calls
    if use_cache and path.exists():
        hit = json.loads(path.read_text(encoding="utf-8"))
        return hit["out"], {**hit["meta"], "cached": True, "latency_s": 0.0}

    payload = {"model": c["model"], "temperature": 0, "max_tokens": max_tokens,
               "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}]}
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    last, t0 = None, time.time()
    for attempt in range(retries):
        try:
            r = requests.post(c["url"], timeout=timeout, json=payload,
                              headers={"Authorization": f"Bearer {c['key']}"})
            r.raise_for_status()
            data = r.json()
            choice = data["choices"][0]
            content = choice["message"]["content"]
            if not content or choice.get("finish_reason") == "length":
                payload["max_tokens"] = int(payload["max_tokens"] * 2)  # reasoning ate budget
                raise ValueError(f"empty/truncated reply (finish={choice.get('finish_reason')})")
            out = _parse(content)
            usage = data.get("usage", {})
            meta = {"model": c["model"], "usage": usage, "cost_usd": cost_of(usage),
                    "latency_s": round(time.time() - t0, 2), "cached": False}
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"out": out, "meta": meta}, ensure_ascii=False),
                            encoding="utf-8")
            return out, meta
        except Exception as e:  # noqa: BLE001 - retried, then surfaced
            last = e
            if attempt + 1 < retries:
                time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last}")
