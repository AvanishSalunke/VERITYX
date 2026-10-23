from dotenv import load_dotenv

load_dotenv()  # must run before langfuse is imported
import os
from pathlib import Path

STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "./storage"))
OPENCODE_API_KEY = os.getenv("OPENCODE_API_KEY", "")
OPENCODE_BASE_URL = os.getenv(
    "OPENCODE_BASE_URL", "https://opencode.ai/inference/openai/v1/chat/completions"
)
OPENCODE_MODEL = os.getenv("OPENCODE_MODEL", "deepseek-v4.1-flash")
# query path uses small Claude models with thinking off (reasoning tokens dominated latency)
CLAUDE_BASE_URL = os.getenv("CLAUDE_BASE_URL", "https://opencode.ai/inference/anthropic")
CONTRACT_MODEL = os.getenv("CONTRACT_MODEL", "claude-haiku-5-5")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "claude-haiku-5-5")
