"""
Central configuration for the agent.

All tunables live here so the rest of the codebase never reaches into
os.environ directly. Makes it trivial to swap providers/models later.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    GROQ_URL: str = "https://api.groq.com/openai/v1/chat/completions"

    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "output")

    # Resilience knobs (used by the retry/backoff wrapper in llm_client.py)
    MAX_RETRIES: int = 3
    BACKOFF_BASE_SECONDS: float = 1.5

    # Guardrails
    MIN_REQUEST_CHARS: int = 8
    MAX_REQUEST_CHARS: int = 4000


settings = Settings()
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
