import json
import time
from typing import Optional, Type, TypeVar

import anthropic
import requests
from pydantic import BaseModel

from . import config
from .tracing import observe

T = TypeVar("T", bound=BaseModel)
_claude: Optional[anthropic.Anthropic] = None


@observe()
def call_json(system: str, user: str, max_tokens: int = 4000, retries: int = 3,
              timeout: float = 90) -> dict:
    """OpenCode chat call returning parsed JSON (same shape the evaluator uses).

    max_tokens only grows on retry when the previous reply was cut off (reasoning models can
    spend the whole budget thinking); other failures retry with the same budget.
    """
    if not config.OPENCODE_API_KEY:
        raise RuntimeError("OPENCODE_API_KEY is not set")
    payload = {
        "model": config.OPENCODE_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {config.OPENCODE_API_KEY}"}
    last = None
    for i in range(retries):
        try:
            r = requests.post(config.OPENCODE_BASE_URL, headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            choice = r.json()["choices"][0]
            content = choice["message"]["content"]
            if not content or choice.get("finish_reason") == "length":
                payload["max_tokens"] *= 2
                raise ValueError("empty or truncated content")
            return json.loads(content.strip().strip("`").removeprefix("json"))
        except Exception as e:  # noqa: BLE001
            last = e
            if i + 1 < retries:
                time.sleep(0.5 * (i + 1))
    raise RuntimeError(f"LLM call failed: {last}")


def claude() -> anthropic.Anthropic:
    """Anthropic SDK client pointed at the OpenCode gateway (same organizer key)."""
    global _claude
    if _claude is None:
        if not config.OPENCODE_API_KEY:
            raise RuntimeError("OPENCODE_API_KEY is not set")
        _claude = anthropic.Anthropic(api_key=config.OPENCODE_API_KEY,
                                      base_url=config.CLAUDE_BASE_URL, max_retries=1, timeout=45)
    return _claude


def _no_thinking(model: str) -> dict:
    # Sonnet 5.5 rejects {"type": "disabled"}; "between_tools" is its thinking-off mode
    if model.startswith("claude-sonnet-5-5"):
        return {"type": "between_tools"}
    return {"type": "disabled"}


@observe(as_type="generation")
def call_claude(system: str, user: str, model: str, max_tokens: int = 2000,
                output_format: Optional[Type[T]] = None):
    """Fast query-path call: thinking off, low effort. Returns text, or a validated
    `output_format` instance when one is given."""
    kwargs = dict(model=model, max_tokens=max_tokens, system=system,
                  thinking=_no_thinking(model), output_config={"effort": "low"},
                  messages=[{"role": "user", "content": user}])
    if output_format is not None:
        msg = claude().messages.parse(output_format=output_format, **kwargs)
        if msg.parsed_output is None:
            raise ValueError(f"no structured output (stop_reason={msg.stop_reason})")
        return msg.parsed_output
    msg = claude().messages.create(**kwargs)
    if msg.stop_reason == "refusal":
        raise RuntimeError("model refused the request")
    return "".join(b.text for b in msg.content if b.type == "text").strip()
