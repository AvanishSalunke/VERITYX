"""Langfuse is mandatory: credentials are read from LANGFUSE_* env vars."""
from . import config  # noqa: F401  (loads .env first)
from langfuse import get_client, observe  # noqa: F401
