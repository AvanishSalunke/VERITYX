import json
import time

import requests

from . import config
from .tracing import observe


@observe()
def call_json(system: str, user: str, max_tokens: int = 4000, retries: int = 3) -> dict:
    """OpenCode chat call returning parsed JSON (same shape the evaluator uses)."""
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
            payload["max_tokens"] = max_tokens * (i + 1)  # reasoning models may exhaust tokens
            r = requests.post(config.OPENCODE_BASE_URL, headers=headers, json=payload, timeout=90)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            if not content:
                raise ValueError("empty content")
            return json.loads(content.strip().strip("`").removeprefix("json"))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"LLM call failed: {last}")
