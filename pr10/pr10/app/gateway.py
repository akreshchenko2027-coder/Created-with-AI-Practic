import logging
import os
import queue
import random
import threading
import time
from dataclasses import replace
from typing import Callable, Iterator

from dotenv import load_dotenv

from .providers.base import Chunk, Provider, ProviderError, Request, Result

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "openai")
FALLBACK = os.getenv("LLM_FALLBACK", "none")
BASE_URL = os.getenv("LLM_BASE_URL")
API_KEY = os.getenv("LLM_API_KEY")

TIMEOUT = float(os.getenv("LLM_TIMEOUT", "20"))
IDLE_TIMEOUT = float(os.getenv("LLM_IDLE_TIMEOUT", "8"))
TOTAL_TIMEOUT = float(os.getenv("LLM_TOTAL_TIMEOUT", "45"))
MAX_ATTEMPTS = int(os.getenv("LLM_MAX_ATTEMPTS", "3"))
BACKOFF_BASE = float(os.getenv("LLM_BACKOFF_BASE", "0.5"))
BACKOFF_MAX = float(os.getenv("LLM_BACKOFF_MAX", "8"))
MAX_CONCURRENT = int(os.getenv("LLM_MAX_CONCURRENT", "2"))

MODEL = os.getenv("LLM_MODEL")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "600"))
logger = logging.getLogger(__name__)
_slots = threading.BoundedSemaphore(MAX_CONCURRENT)

if min(TIMEOUT, IDLE_TIMEOUT, TOTAL_TIMEOUT, BACKOFF_BASE, BACKOFF_MAX) <= 0:
    raise ValueError("Таймаути та паузи шлюзу мають бути додатними.")
if MAX_ATTEMPTS < 1 or MAX_CONCURRENT < 1 or MAX_TOKENS < 1:
    raise ValueError("Кількість спроб, одночасних запитів і токенів має бути додатною.")


def settings() -> dict:
    """Чинні налаштування шлюзу — для сторінки й журналу. Ключа тут
    немає і бути не може."""
    return {
        "provider": PROVIDER,
        "fallback": FALLBACK,
        "model": MODEL,
        "timeout": TIMEOUT,
        "idle_timeout": IDLE_TIMEOUT,
        "total_timeout": TOTAL_TIMEOUT,
        "max_attempts": MAX_ATTEMPTS,
        "max_concurrent": MAX_CONCURRENT,
    }


def make_provider(name: str) -> Provider:
    if name == "mock":
        from .providers.mock import MockProvider

        return MockProvider()
    if name == "openai" or name == MODEL or name == FALLBACK:
        if not (BASE_URL and API_KEY and MODEL):
            raise ProviderError("auth", "Справжній провайдер не налаштований.")
        from .providers.openai_compat import OpenAICompatProvider

        model = MODEL if name == "openai" else name
        provider = OpenAICompatProvider(BASE_URL, API_KEY, model, TIMEOUT)
        provider.name = "openai" if name == "openai" else name
        return provider
    raise ProviderError("bad_request", "Невідомий провайдер.")


def _cancelled(is_cancelled: Callable[[], bool], stop: threading.Event) -> bool:
    return stop.is_set() or is_cancelled()


def _wait(delay: float, deadline: float, is_cancelled: Callable[[], bool]) -> None:
    end = min(time.monotonic() + delay, deadline)
    while True:
        if is_cancelled():
            raise ProviderError("cancelled", "Генерацію скасовано.")
        remaining = end - time.monotonic()
        if remaining <= 0:
            if time.monotonic() >= deadline:
                raise ProviderError("timeout", "Вичерпано загальний час запиту.", retryable=True)
            return
        time.sleep(min(0.05, remaining))


def _stream_attempt(
    provider: Provider,
    request: Request,
    on_delta: Callable[[str], None],
    is_cancelled: Callable[[], bool],
    total_deadline: float,
) -> tuple[Result, bool]:
    attempt_started = time.monotonic()
    attempt_deadline = min(attempt_started + TIMEOUT, total_deadline)
    stop = threading.Event()
    chunks: queue.Queue[tuple[str, object]] = queue.Queue()
    metadata = {
        **request.metadata,
        "attempt_timeout": max(0.01, attempt_deadline - attempt_started),
        "idle_timeout": min(IDLE_TIMEOUT, max(0.01, attempt_deadline - attempt_started)),
        "is_cancelled": lambda: _cancelled(is_cancelled, stop),
    }
    attempt_request = replace(
        request,
        model=getattr(provider, "model", request.model),
        metadata=metadata,
    )

    def produce() -> None:
        try:
            for chunk in provider.stream(attempt_request):
                if _cancelled(is_cancelled, stop):
                    break
                chunks.put(("chunk", chunk))
            chunks.put(("end", None))
        except ProviderError as exc:
            chunks.put(("error", exc))
        except Exception as exc:
            logger.exception("Provider %s failed while streaming", provider.name)
            chunks.put(("error", ProviderError(
                "unknown", "Провайдер не зміг завершити генерацію.", detail=type(exc).__name__
            )))

    thread = threading.Thread(target=produce, name="ai-provider-stream", daemon=True)
    thread.start()
    text: list[str] = []
    usage = None
    finish_reason = None
    last_chunk = attempt_started
    try:
        while True:
            if is_cancelled():
                stop.set()
                raise ProviderError("cancelled", "Генерацію скасовано.")
            now = time.monotonic()
            if now >= total_deadline:
                stop.set()
                raise ProviderError("timeout", "Вичерпано загальний час запиту.", retryable=True)
            if now >= attempt_deadline:
                stop.set()
                raise ProviderError("timeout", "Час очікування спроби вичерпано.", retryable=True)
            if now - last_chunk >= IDLE_TIMEOUT:
                stop.set()
                raise ProviderError("timeout", "Потік провайдера надто довго не надсилав фрагментів.",
                                    retryable=True)
            wait_for = min(0.1, total_deadline - now, attempt_deadline - now,
                           IDLE_TIMEOUT - (now - last_chunk))
            try:
                kind, value = chunks.get(timeout=max(0.001, wait_for))
            except queue.Empty:
                continue
            last_chunk = time.monotonic()
            if kind == "error":
                raise value
            if kind == "end":
                result = Result(
                    "".join(text),
                    provider.name,
                    getattr(provider, "model", request.model or MODEL or "unknown"),
                    usage,
                    finish_reason,
                    time.monotonic() - attempt_started,
                )
                return result, bool(text)
            chunk = value
            if not isinstance(chunk, Chunk):
                raise ProviderError("unknown", "Провайдер повернув некоректний фрагмент.")
            if chunk.usage:
                usage = chunk.usage
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
            if chunk.text:
                text.append(chunk.text)
                on_delta(chunk.text)
    finally:
        stop.set()


def generate(request: Request, on_delta: Callable[[str], None],
             on_status: Callable[[dict], None], is_cancelled: Callable[[], bool]) -> Result:
    started = time.monotonic()
    deadline = started + TOTAL_TIMEOUT
    emitted = False

    def delta(text: str) -> None:
        nonlocal emitted
        emitted = True
        on_delta(text)

    def acquire_slot(name: str, attempt: int, fallback: bool) -> None:
        announced = False
        while True:
            if is_cancelled():
                raise ProviderError("cancelled", "Генерацію скасовано.")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProviderError("timeout", "Вичерпано загальний час запиту.", retryable=True)
            if _slots.acquire(timeout=min(0.1, remaining)):
                return
            if not announced:
                on_status({
                    "status": "running",
                    "attempt": attempt,
                    "provider": name,
                    "fallback": fallback,
                    "detail": "очікування вільного місця",
                })
                announced = True

    def run_provider(name: str, attempt: int, fallback: bool) -> tuple[Result, bool]:
        if is_cancelled():
            raise ProviderError("cancelled", "Генерацію скасовано.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProviderError("timeout", "Вичерпано загальний час запиту.", retryable=True)
        provider = make_provider(name)
        on_status({
            "status": "running",
            "attempt": attempt,
            "provider": provider.name,
            "fallback": fallback,
            "detail": "запасний варіант" if fallback else "",
        })
        acquire_slot(provider.name, attempt, fallback)
        try:
            return _stream_attempt(provider, request, delta, is_cancelled, deadline)
        finally:
            _slots.release()

    error: ProviderError | None = None
    attempts = 0
    provider_name = PROVIDER
    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            attempts = attempt
            try:
                result, _ = run_provider(PROVIDER, attempt, False)
                return replace(result, elapsed=time.monotonic() - started)
            except ProviderError as exc:
                if exc.kind == "cancelled" or is_cancelled():
                    raise ProviderError("cancelled", "Генерацію скасовано.") from exc
                error = exc
                if emitted or not exc.retryable or attempt >= MAX_ATTEMPTS:
                    break
                remaining = deadline - time.monotonic()
                backoff = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** (attempt - 1))) * random.uniform(0.5, 1.5)
                delay = exc.retry_after if exc.retry_after is not None else backoff
                if delay >= remaining:
                    on_status({
                        "status": "running",
                        "attempt": attempt,
                        "provider": PROVIDER,
                        "fallback": False,
                        "detail": "повтор не вміщується у бюджет часу",
                    })
                    break
                on_status({
                    "status": "running",
                    "attempt": attempt + 1,
                    "provider": PROVIDER,
                    "fallback": False,
                    "detail": f"повтор через {delay:.1f} с",
                })
                _wait(delay, deadline, is_cancelled)

        fallback_available = (
            FALLBACK != "none"
            and FALLBACK != PROVIDER
            and not (PROVIDER == "openai" and FALLBACK == MODEL)
        )
        if error is not None and not emitted and fallback_available:
            if is_cancelled():
                raise ProviderError("cancelled", "Генерацію скасовано.")
            provider_name = FALLBACK
            on_status({
                "status": "running",
                "attempt": attempts + 1,
                "provider": FALLBACK,
                "fallback": True,
                "detail": "основний провайдер не відповів",
            })
            try:
                result, _ = run_provider(FALLBACK, attempts + 1, True)
                return replace(result, elapsed=time.monotonic() - started)
            except ProviderError as fallback_error:
                if fallback_error.kind == "cancelled" or is_cancelled():
                    raise ProviderError("cancelled", "Генерацію скасовано.") from fallback_error
                error = fallback_error

        if error is None:
            error = ProviderError("unknown", "Не вдалося створити чернетку.")
        if emitted and error.retryable:
            error = ProviderError(
                error.kind,
                "Потік перервався після частини тексту. Чернетка збережена для перевірки; повторіть запит, щоб почати заново.",
                retryable=True,
                retry_after=error.retry_after,
                detail=error.detail,
            )
        raise error
    except ProviderError:
        raise
    except Exception as exc:
        logger.exception("Gateway failed for provider %s", provider_name)
        raise ProviderError("unknown", "Не вдалося створити чернетку.", detail=type(exc).__name__) from exc
