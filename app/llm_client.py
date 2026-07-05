"""
Thin wrapper around Groq's OpenAI-compatible /chat/completions endpoint.

Groq is used because it has a genuinely free tier and is fast enough for a
live demo. Swapping to Ollama/LM Studio/Gemini only requires changing
`_call_groq` (or pointing GROQ_URL at an OpenAI-compatible local server) --
nothing else in the agent depends on the provider.

Resilience: every call is wrapped in retry-with-exponential-backoff so a
transient 429/5xx doesn't kill the whole pipeline. If every retry fails
(e.g. no API key was configured for this demo run), we fall back to a
deterministic offline generator so the API never hard-crashes -- it degrades
to template output instead of a 500.
"""
import json
import re
import time
import logging
import requests

from app.config import settings

logger = logging.getLogger("agent.llm")


class LLMError(Exception):
    pass


class RateLimitError(LLMError):
    """Raised on HTTP 429. Carries the provider's own suggested wait time
    (in seconds) so the retry loop can wait the *correct* amount instead of
    guessing with fixed exponential backoff, which is what was causing
    retries to burn out before Groq's per-minute window actually reset."""
    def __init__(self, message: str, retry_after: float):
        super().__init__(message)
        self.retry_after = retry_after


_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)s", re.IGNORECASE)


def _call_groq(messages: list[dict], json_mode: bool = False, temperature: float = 0.4) -> str:
    if not settings.GROQ_API_KEY:
        raise LLMError("GROQ_API_KEY not configured")

    headers = {
        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1200,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    resp = requests.post(settings.GROQ_URL, headers=headers, json=payload, timeout=30)

    if resp.status_code == 429:
        # Prefer the header if Groq sends one; otherwise parse it out of the
        # error message body, e.g. "...Please try again in 5.31s...".
        retry_after = resp.headers.get("retry-after")
        if retry_after is None:
            match = _RETRY_AFTER_RE.search(resp.text)
            retry_after = match.group(1) if match else "2"
        raise RateLimitError(f"Groq API returned 429: {resp.text[:300]}", retry_after=float(retry_after))

    if resp.status_code != 200:
        raise LLMError(f"Groq API returned {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    return data["choices"][0]["message"]["content"]


def chat(messages: list[dict], json_mode: bool = False, temperature: float = 0.4) -> str:
    """
    Call the LLM with retry.
      - On a 429 (rate limit), wait exactly what Groq tells us to wait
        (plus a small safety buffer), since guessing with fixed backoff
        was retrying *before* the provider's window had actually reset.
      - On any other transient error, fall back to exponential backoff.
    Raises LLMError if all retries are exhausted -- caller decides on fallback.
    """
    last_err = None
    for attempt in range(1, settings.MAX_RETRIES + 1):
        try:
            return _call_groq(messages, json_mode=json_mode, temperature=temperature)
        except RateLimitError as e:
            last_err = e
            wait = e.retry_after + 0.5
            logger.warning(f"Rate limited on attempt {attempt}. Waiting {wait:.1f}s (provider-reported).")
            if attempt < settings.MAX_RETRIES:
                time.sleep(wait)
        except (LLMError, requests.RequestException) as e:
            last_err = e
            wait = settings.BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            logger.warning(f"LLM call attempt {attempt} failed: {e}. Retrying in {wait:.1f}s")
            if attempt < settings.MAX_RETRIES:
                time.sleep(wait)
    raise LLMError(f"All {settings.MAX_RETRIES} attempts failed. Last error: {last_err}")


def safe_json_parse(text: str) -> dict:
    """
    LLMs occasionally wrap JSON in markdown fences or add stray prose.
    Strip that defensively before parsing, and raise a clear error otherwise.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # last resort: grab the substring between the first { and last }
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass
        raise LLMError(f"Could not parse JSON from LLM output: {e}\nRaw: {cleaned[:300]}")
