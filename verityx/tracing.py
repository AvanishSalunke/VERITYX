"""Langfuse telemetry (SDK v5, OpenTelemetry-based).

Every /query becomes one Langfuse trace with nested observations (planner, retrieval, answer
generation, verifier); every LLM call is a `generation` carrying model, token usage, cost and
latency. Indexing is traced the same way. Without LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY
(or with LANGFUSE_DISABLED=1) all of this is a no-op, so the pipeline runs unchanged.
"""
import contextlib
import os

_client = None


def enabled():
    return (bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
            and os.getenv("LANGFUSE_DISABLED", "0") != "1")


def client():
    global _client
    if _client is None and enabled():
        from langfuse import get_client
        _client = get_client()
    return _client


class _Noop:
    def update(self, **_):
        pass


@contextlib.contextmanager
def observation(name, as_type="span", **fields):
    """Open a (nested) Langfuse observation; yields an object with .update(**fields)."""
    c = client()
    if c is None:
        yield _Noop()
        return
    with c.start_as_current_observation(name=name, as_type=as_type, **fields) as obs:
        yield obs


@contextlib.contextmanager
def trace(name, session_id=None, metadata=None, tags=None, **fields):
    """Root observation that also names the trace and tags it (collection as session)."""
    c = client()
    if c is None:
        yield _Noop()
        return
    from langfuse import propagate_attributes
    meta = {k: str(v)[:200] for k, v in (metadata or {}).items()}
    with propagate_attributes(trace_name=name, session_id=session_id, metadata=meta, tags=tags):
        with c.start_as_current_observation(name=name, as_type="span", **fields) as obs:
            yield obs


def bind(fn):
    """Carry the current trace context into a worker thread (ThreadPoolExecutor does not), so
    observations made there nest under the caller's trace instead of starting new ones."""
    if client() is None:
        return fn
    from opentelemetry import context
    ctx = context.get_current()

    def wrapped(*args, **kwargs):
        token = context.attach(ctx)
        try:
            return fn(*args, **kwargs)
        finally:
            context.detach(token)
    return wrapped


def current_trace_url():
    c = client()
    try:
        return c.get_trace_url() if c else None
    except Exception:  # noqa: BLE001 - a URL is a convenience, never an error
        return None


def flush():
    c = client()
    if c is not None:
        c.flush()


def usage_details(usage):
    """OpenAI-style usage -> Langfuse usage_details."""
    out = {"input": usage.get("prompt_tokens", 0), "output": usage.get("completion_tokens", 0),
           "total": usage.get("total_tokens", 0)}
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    if reasoning:
        out["reasoning"] = reasoning
    if usage.get("cache_read_input_tokens"):
        out["cache_read_input"] = usage["cache_read_input_tokens"]
    return out


def clip(text, n=6000):
    """Keep traced payloads readable; full prompts can be tens of thousands of tokens."""
    text = str(text)
    return text if len(text) <= n else text[:n] + f" ...[+{len(text) - n} chars]"
