"""Static (model2vec) embeddings: the whole training corpus embeds in ~2s on CPU, versus
8-37 min for a transformer encoder on the same machine, which keeps /index fast."""
import os
import threading

import numpy as np

MODEL = os.getenv("EMBED_MODEL", "minishlab/potion-base-8M")
_model = None
_lock = threading.Lock()   # the HF tokenizer inside model2vec is not thread-safe


def _get():
    global _model
    if _model is None:
        from model2vec import StaticModel
        _model = StaticModel.from_pretrained(MODEL)
    return _model


def embed(texts):
    with _lock:
        v = np.asarray(_get().encode(list(texts), batch_size=1024), dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v
