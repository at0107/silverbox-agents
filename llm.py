"""Gemini access: reads settings from the environment and retries on rate limits.

Nothing in this file ever prints or stores the API key.
"""
import os
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


def load_dotenv_if_present(path=".env"):
    """Tiny .env reader: KEY=VALUE lines go into os.environ unless already set.
    (.env is git-ignored. The values are never printed.)"""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_dotenv_if_present()

# This machine has no working IPv6: Python tries the IPv6 address of googleapis.com first and
# hangs in connect(). Resolve host names to IPv4 only (set USE_IPV6=1 to turn this off).
if not os.environ.get("USE_IPV6"):
    import socket
    _orig_getaddrinfo = socket.getaddrinfo
    socket.getaddrinfo = lambda host, *a, **k: [
        r for r in _orig_getaddrinfo(host, *a, **k) if r[0] == socket.AF_INET]


def require_key():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY is not set (export it, or put it in a .env file).")
    return key


def make_llm(temperature=0.2):
    """Chat model whose name comes from GEMINI_MODEL."""
    model = os.environ.get("GEMINI_MODEL")
    if not model:
        raise SystemExit("GEMINI_MODEL is not set (run check_gemini.py to see the options).")
    # max_retries=0: we do our own retry/backoff below so that it is visible and controlled
    return ChatGoogleGenerativeAI(model=model, google_api_key=require_key(),
                                  temperature=temperature, max_retries=0)


def _is_retryable(exc):
    """429 (rate limit) and 5xx 'try again later' errors (e.g. 503 'high demand') are temporary."""
    text = f"{type(exc).__name__} {exc}"
    if "PerDay" in text:       # daily free-tier quota used up: waiting minutes cannot fix it
        return False
    return any(t in text for t in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE",
                                   "500 INTERNAL", "504", "DEADLINE_EXCEEDED")) \
        or "rate limit" in text.lower()


# Retry only those temporary errors: wait 4s, 8s, 16s, ... up to 60s, at most 14 attempts
# (about 12 minutes in total). Other errors (404 wrong model, bad key, bugs) fail at once.
with_backoff = retry(retry=retry_if_exception(_is_retryable),
                     wait=wait_exponential(multiplier=2, min=4, max=60),
                     stop=stop_after_attempt(14), reraise=True,
                     before_sleep=lambda rs: print(
                         f"  (temporary API error, retry {rs.attempt_number} in a moment)", flush=True))


@with_backoff
def call(runnable, messages):
    """Invoke a chat model (or a structured-output chain) with backoff on temporary errors."""
    return runnable.invoke(messages)
