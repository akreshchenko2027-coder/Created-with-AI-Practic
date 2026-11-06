import os
import threading
import time

from dotenv import load_dotenv

load_dotenv()

PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
WINDOW = 60.0
_requests: dict[str, list[float]] = {}
_lock = threading.Lock()

if PER_MINUTE < 1:
    raise ValueError("RATE_LIMIT_PER_MINUTE має бути не меншим за 1.")


class RateLimited(Exception):
    """Ліміт вичерпано. `retry_after` — через скільки секунд пробувати."""

    def __init__(self, retry_after: float) -> None:
        super().__init__(f"ліміт вичерпано, повторіть через {retry_after:.0f} с")
        self.retry_after = retry_after


def settings() -> dict:
    """Чинні обмеження — для сторінки."""
    return {"per_minute": PER_MINUTE}


def acquire(client: str) -> None:
    key = client.strip()[:120] or "anonymous"
    now = time.monotonic()
    cutoff = now - WINDOW
    with _lock:
        for name in tuple(_requests):
            recent = [stamp for stamp in _requests[name] if stamp > cutoff]
            if recent:
                _requests[name] = recent
            else:
                del _requests[name]
        recent = _requests.setdefault(key, [])
        if len(recent) >= PER_MINUTE:
            raise RateLimited(max(1, recent[0] + WINDOW - now))
        recent.append(now)
