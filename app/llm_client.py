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

Rate-limit visibility: Groq returns x-ratelimit-* headers on *every*
response (success or failure). We capture the latest values in a
module-level snapshot so /health and /agent can expose real remaining/used
numbers to the frontend instead of guessing.
"""
from __future__ import annotations  # lets "str | None" etc. run on Python 3.9

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


# If Groq reports a wait longer than this, it's a daily/hourly quota issue,
# not a short burst limit -- don't block a live request for minutes on the
# hope it clears; fail fast so the caller's fallback/placeholder logic can
# kick in immediately instead of the pipeline appearing to hang.
MAX_ACCEPTABLE_RETRY_WAIT_SECONDS = 20.0


_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)s", re.IGNORECASE)

# Groq's reset header is a duration string like "2m59.56s", not a plain
# number of seconds -- this parses hours/minutes/seconds, any of which may
# be absent.
_DURATION_RE = re.compile(r"(?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?")

# Last-known rate-limit snapshot, updated on every Groq response (success or
# 429). /health and /agent read this via get_rate_limit_info() to show the
# frontend real numbers instead of "not reported."
_last_rate_limit: dict = {}


def _parse_duration(s: str | None) -> float | None:
    if not s:
        return None
    s = s.strip()
    if s.replace(".", "", 1).isdigit():
        return float(s)
    m = _DURATION_RE.fullmatch(s)
    if not m or not any(m.groups()):
        return None
    h, mn, sec = m.groups()
    return (int(h or 0) * 3600) + (int(mn or 0) * 60) + float(sec or 0)


def _capture_rate_limit_headers(resp: requests.Response) -> None:
    """Groq sends these on every request/response, not just on 429s:
      x-ratelimit-limit-requests, x-ratelimit-remaining-requests,
      x-ratelimit-reset-requests (and the -tokens equivalents).
    We track the *requests* counters since that's what actually throttles
    this pipeline (one call per section)."""
    global _last_rate_limit
    remaining = resp.headers.get("x-ratelimit-remaining-requests")
    limit = resp.headers.get("x-ratelimit-limit-requests")
    reset = resp.headers.get("x-ratelimit-reset-requests")

    remaining_tokens = resp.headers.get("x-ratelimit-remaining-tokens")
    limit_tokens = resp.headers.get("x-ratelimit-limit-tokens")
    reset_tokens = resp.headers.get("x-ratelimit-reset-tokens")

    if remaining is None and limit is None:
        return  # headers absent on this response; keep last-known snapshot

    _last_rate_limit = {
        "remaining": int(remaining) if remaining is not None else None,
        "limit": int(limit) if limit is not None else None,
        "reset_seconds": _parse_duration(reset),
        "remaining_tokens": int(remaining_tokens) if remaining_tokens is not None else None,
        "limit_tokens": int(limit_tokens) if limit_tokens is not None else None,
        "reset_tokens_seconds": _parse_duration(reset_tokens),
        "updated_at": time.time(),
    }


def get_rate_limit_info() -> dict:
    """Read-only snapshot for /health and /agent to expose to the frontend.
    Empty dict until the first real Groq call has been made."""
    return dict(_last_rate_limit)


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
    _capture_rate_limit_headers(resp)

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
            if e.retry_after > MAX_ACCEPTABLE_RETRY_WAIT_SECONDS:
                # This is a daily/hourly quota exhaustion, not a short burst
                # limit -- don't block the request for minutes. Fail now so
                # the caller's fallback path (placeholder text / fallback
                # plan) kicks in immediately instead of the UI hanging.
                logger.error(
                    f"Rate limited on attempt {attempt} with a {e.retry_after:.0f}s "
                    f"wait -- exceeds the {MAX_ACCEPTABLE_RETRY_WAIT_SECONDS:.0f}s "
                    f"cap, likely a daily/hourly quota. Failing fast instead of blocking."
                )
                raise
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