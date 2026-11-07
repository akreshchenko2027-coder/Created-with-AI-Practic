import json
import logging
import os
import queue
import random
import threading
import time
from dataclasses import replace
from typing import Callable

from dotenv import load_dotenv

from . import budget, cache, costs, drafting
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

logger = logging.getLogger("app.gateway")
_slots = threading.BoundedSemaphore(MAX_CONCURRENT)

if min(TIMEOUT, IDLE_TIMEOUT, TOTAL_TIMEOUT, BACKOFF_BASE, BACKOFF_MAX) <= 0:
    raise ValueError("Таймаути та паузи шлюзу мають бути додатними.")
if MAX_ATTEMPTS < 1 or MAX_CONCURRENT < 1 or MAX_TOKENS < 1:
    raise ValueError("Кількість спроб, одночасних запитів і токенів має бути додатною.")


def settings() -> dict:
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
    if name in ("openai", "ollama") or name == MODEL or name == FALLBACK:
        if not (BASE_URL and MODEL and (API_KEY or name == "ollama")):
            raise ProviderError("auth", "Справжній провайдер не налаштований.")
        from .providers.openai_compat import OpenAICompatProvider

        model = MODEL if name in ("openai", "ollama") else name
        provider = OpenAICompatProvider(BASE_URL, API_KEY or "ollama", model, TIMEOUT)
        provider.name = name
        return provider
    raise ProviderError("bad_request", "Невідомий провайдер.")


def _prompt_text(request: Request) -> str:
    return json.dumps(request.messages, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _usage_totals(records: list[dict]) -> dict | None:
    if not records or any(record["usage"] is None for record in records):
        return None
    prompt = sum(record["usage"].get("prompt_tokens", 0) for record in records)
    completion = sum(record["usage"].get("completion_tokens", 0) for record in records)
    cached = sum(
        (record["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        for record in records
    )
    result = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
    if cached:
        result["prompt_tokens_details"] = {"cached_tokens": cached}
    return result


def _log_attempt(request: Request, provider: Provider, attempt: int, fallback: bool,
                 queue_ms: float, duration_ms: float, first_token_ms: float | None,
                 completion_chars: int, status: str, kind: str | None, usage: dict | None,
                 cost: float | None, estimated: bool, cache_key: str) -> None:
    prompt_version = request.metadata.get("prompt_version", "unknown")
    logger.log(logging.WARNING if status != "ok" else logging.INFO, "llm.attempt", extra={
        "request_id": request.metadata.get("http_request_id"),
        "draft_id": request.metadata.get("draft_id"),
        "source_request_id": request.metadata.get("request_id"),
        "client": request.metadata.get("client"),
        "attempt": attempt,
        "provider": provider.name,
        "model": getattr(provider, "model", request.model),
        "prompt_version": prompt_version,
        "queue_ms": round(queue_ms, 1),
        "duration_ms": round(duration_ms, 1),
        "first_token_ms": round(first_token_ms, 1) if first_token_ms is not None else None,
        "status": status,
        "kind": kind,
        "fallback": fallback,
        "cached": False,
        "prompt_chars": sum(len(str(message.get("content", ""))) for message in request.messages),
        "completion_chars": completion_chars,
        "prompt_tokens": usage.get("prompt_tokens") if usage else None,
        "completion_tokens": usage.get("completion_tokens") if usage else None,
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) if usage else 0,
        "cost": cost,
        "cost_estimated": estimated,
        "cache_key": cache_key,
    })


def _stream_attempt(
    provider: Provider,
    request: Request,
    on_delta: Callable[[str], None],
    is_cancelled: Callable[[], bool],
    total_deadline: float,
    attempt: int,
    fallback: bool,
    queue_ms: float,
    cache_key: str,
) -> Result:
    prompt = _prompt_text(request)
    expected_cost = costs.maximum_cost(prompt, request.max_tokens, provider=provider.name)
    try:
        reservation = budget.check(
            expected_cost,
            request_id=request.metadata.get("http_request_id"),
            draft_id=request.metadata.get("draft_id"),
        )
    except budget.BudgetExceeded as exc:
        logger.error("llm.attempt", extra={
            "request_id": request.metadata.get("http_request_id"),
            "draft_id": request.metadata.get("draft_id"),
            "source_request_id": request.metadata.get("request_id"),
            "client": request.metadata.get("client"),
            "attempt": attempt,
            "provider": provider.name,
            "model": getattr(provider, "model", request.model),
            "prompt_version": request.metadata.get("prompt_version", "unknown"),
            "queue_ms": round(queue_ms, 1),
            "duration_ms": 0.0,
            "first_token_ms": None,
            "status": "rejected",
            "kind": "budget",
            "fallback": fallback,
            "cached": False,
            "cost": 0.0,
            "cost_estimated": False,
            "cache_key": cache_key,
        })
        raise ProviderError("budget", str(exc), detail="BudgetExceeded") from exc

    started = time.monotonic()
    deadline = min(started + TIMEOUT, total_deadline)
    stop = threading.Event()
    chunks: queue.Queue[tuple[str, object]] = queue.Queue()
    metadata = {
        **request.metadata,
        "attempt_timeout": max(0.01, deadline - started),
        "idle_timeout": min(IDLE_TIMEOUT, max(0.01, deadline - started)),
        "is_cancelled": lambda: stop.is_set() or is_cancelled(),
    }
    attempt_request = replace(request, model=getattr(provider, "model", request.model), metadata=metadata)
    text: list[str] = []
    usage = None
    finish_reason = None
    first_token_ms = None
    last_chunk = started
    recorded = False

    def account(status: str, error: ProviderError | None = None) -> tuple[float | None, bool]:
        nonlocal recorded
        amount, estimated = costs.estimate(
            usage, provider=provider.name, model=getattr(provider, "model", request.model),
            text="".join(text), prompt=prompt,
        )
        budget.record(amount, reservation)
        recorded = True
        _log_attempt(
            request, provider, attempt, fallback, queue_ms,
            (time.monotonic() - started) * 1000, first_token_ms,
            sum(map(len, text)), status, error.kind if error else None,
            usage, amount, estimated, cache_key,
        )
        return amount, estimated

    def produce() -> None:
        try:
            for chunk in provider.stream(attempt_request):
                if stop.is_set() or is_cancelled():
                    break
                chunks.put(("chunk", chunk))
            chunks.put(("end", None))
        except ProviderError as exc:
            chunks.put(("error", exc))
        except Exception as exc:
            logger.error("provider.stream_error", extra={
                "request_id": request.metadata.get("http_request_id"),
                "draft_id": request.metadata.get("draft_id"),
                "provider": provider.name,
                "error_type": type(exc).__name__,
            })
            chunks.put(("error", ProviderError(
                "unknown", "Провайдер не зміг завершити генерацію.", detail=type(exc).__name__
            )))

    thread = threading.Thread(target=produce, name="ai-provider-stream", daemon=True)
    try:
        thread.start()
        while True:
            if is_cancelled():
                raise ProviderError("cancelled", "Генерацію скасовано.")
            now = time.monotonic()
            if now >= total_deadline:
                raise ProviderError("timeout", "Вичерпано загальний час запиту.", retryable=True)
            if now >= deadline:
                raise ProviderError("timeout", "Час очікування спроби вичерпано.", retryable=True)
            if now - last_chunk >= IDLE_TIMEOUT:
                raise ProviderError("timeout", "Потік провайдера надто довго не надсилав фрагментів.",
                                    retryable=True)
            wait_for = min(0.1, total_deadline - now, deadline - now,
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
                    time.monotonic() - started,
                )
                amount, estimated = account("ok")
                return replace(result, cost=amount, cost_estimated=estimated)
            chunk = value
            if not isinstance(chunk, Chunk):
                raise ProviderError("unknown", "Провайдер повернув некоректний фрагмент.")
            if chunk.usage:
                usage = chunk.usage
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
            if chunk.text:
                if first_token_ms is None:
                    first_token_ms = (time.monotonic() - started) * 1000
                text.append(chunk.text)
                on_delta(chunk.text)
    except ProviderError as exc:
        if not recorded:
            amount, estimated = account("failed", exc)
            exc.cost = amount
            exc.cost_estimated = estimated
            exc.usage = usage
            exc.provider = provider.name
            exc.model = getattr(provider, "model", request.model)
        raise
    except Exception as exc:
        error = ProviderError("unknown", "Не вдалося завершити звертання до провайдера.",
                              detail=type(exc).__name__)
        if not recorded:
            amount, estimated = account("failed", error)
            error.cost = amount
            error.cost_estimated = estimated
            error.usage = usage
            error.provider = provider.name
            error.model = getattr(provider, "model", request.model)
        raise error from exc
    finally:
        stop.set()


def generate(request: Request, on_delta: Callable[[str], None],
             on_status: Callable[[dict], None], is_cancelled: Callable[[], bool]) -> Result:
    started = time.monotonic()
    deadline = started + TOTAL_TIMEOUT
    emitted = False
    prompt_version = str(request.metadata.get("prompt_version") or drafting.PROMPT_VERSION)
    cache_key = cache.make_key(request, prompt_version)
    fresh = bool(request.metadata.get("fresh"))
    request_id = request.metadata.get("http_request_id")
    draft_id = request.metadata.get("draft_id")
    common = {
        "request_id": request_id,
        "draft_id": draft_id,
        "source_request_id": request.metadata.get("request_id"),
        "client": request.metadata.get("client"),
        "prompt_version": prompt_version,
        "cache_key": cache_key,
    }
    if fresh:
        cache.note_bypassed()
        logger.info("cache.bypass", extra={**common, "reason": "fresh"})
    else:
        cached = cache.get(cache_key)
        if cached is not None:
            tokens = (cached.usage or {}).get("total_tokens", 0)
            if not tokens and cached.usage:
                tokens = (cached.usage.get("prompt_tokens", 0)
                          + cached.usage.get("completion_tokens", 0))
            on_status({
                "status": "running",
                "attempt": 0,
                "provider": cached.provider,
                "model": cached.model,
                "cached": True,
                "queue_ms": 0.0,
                "detail": "відповідь узято з кешу",
            })
            if cached.text:
                on_delta(cached.text)
            logger.info("cache.hit", extra={
                **common,
                "provider": cached.provider,
                "model": cached.model,
                "cost_saved": cached.cost,
                "tokens_saved": tokens,
            })
            return replace(
                cached, usage=None, cost=0.0, cost_estimated=False, cached=True,
                cache_saved_cost=cached.cost, cache_saved_tokens=tokens,
                elapsed=time.monotonic() - started,
            )
        logger.info("cache.miss", extra={**common})

    def delta(text: str) -> None:
        nonlocal emitted
        emitted = True
        on_delta(text)

    def acquire_slot(name: str, attempt: int, fallback: bool) -> float:
        queue_started = time.monotonic()
        announced = False
        while True:
            if is_cancelled():
                raise ProviderError("cancelled", "Генерацію скасовано.")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProviderError("timeout", "Вичерпано загальний час запиту.", retryable=True)
            if _slots.acquire(timeout=min(0.1, remaining)):
                return (time.monotonic() - queue_started) * 1000
            if not announced:
                on_status({
                    "status": "running",
                    "attempt": attempt,
                    "provider": name,
                    "fallback": fallback,
                    "queue_ms": (time.monotonic() - queue_started) * 1000,
                    "detail": "очікування вільного місця",
                })
                announced = True

    def run_provider(name: str, attempt: int, fallback: bool) -> Result:
        if is_cancelled():
            raise ProviderError("cancelled", "Генерацію скасовано.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProviderError("timeout", "Вичерпано загальний час запиту.", retryable=True)
        try:
            provider = make_provider(name)
        except ProviderError as exc:
            logger.warning("llm.attempt", extra={
                **common,
                "attempt": attempt,
                "provider": name,
                "model": MODEL,
                "prompt_version": prompt_version,
                "queue_ms": 0.0,
                "duration_ms": 0.0,
                "first_token_ms": None,
                "status": "failed",
                "kind": exc.kind,
                "fallback": fallback,
                "cached": False,
                "cost": 0.0,
                "cost_estimated": False,
            })
            raise
        on_status({
            "status": "running",
            "attempt": attempt,
            "provider": provider.name,
            "model": getattr(provider, "model", request.model),
            "fallback": fallback,
            "detail": "запасний варіант" if fallback else "",
        })
        queue_ms = acquire_slot(provider.name, attempt, fallback)
        try:
            return _stream_attempt(
                provider, request, delta, is_cancelled, deadline, attempt, fallback, queue_ms, cache_key
            )
        finally:
            _slots.release()

    failures: list[dict] = []
    error: ProviderError | None = None
    attempts = 0
    provider_name = PROVIDER

    def save_failure(exc: ProviderError) -> None:
        failures.append({
            "cost": exc.cost,
            "cost_estimated": exc.cost_estimated,
            "usage": exc.usage,
        })

    def finish_result(result: Result, fallback: bool) -> Result:
        records = [*failures, {
            "cost": result.cost,
            "cost_estimated": result.cost_estimated,
            "usage": result.usage,
        }]
        cost_values = [item["cost"] for item in records]
        total_cost = round(sum(value for value in cost_values if value is not None), 12)
        complete_cost = total_cost if all(value is not None for value in cost_values) else None
        total_usage = _usage_totals(records)
        final = replace(
            result,
            usage=total_usage,
            cost=complete_cost,
            cost_estimated=any(item["cost_estimated"] for item in records),
            elapsed=time.monotonic() - started,
        )
        if not fallback:
            cache.put(cache_key, final)
        return final

    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            attempts = attempt
            try:
                return finish_result(run_provider(PROVIDER, attempt, False), False)
            except ProviderError as exc:
                if exc.kind == "cancelled" or is_cancelled():
                    raise ProviderError("cancelled", "Генерацію скасовано.", attempts=attempts) from exc
                error = exc
                if exc.kind == "budget" or exc.cost is not None:
                    save_failure(exc)
                if emitted or not exc.retryable or attempt >= MAX_ATTEMPTS:
                    break
                remaining = deadline - time.monotonic()
                backoff = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** (attempt - 1))) * random.uniform(0.5, 1.5)
                delay = exc.retry_after if exc.retry_after is not None else backoff
                if delay >= remaining:
                    break
                logger.warning("llm.retry", extra={
                    **common,
                    "attempt": attempt,
                    "next_attempt": attempt + 1,
                    "kind": exc.kind,
                    "delay_ms": round(delay * 1000, 1),
                    "provider": PROVIDER,
                })
                on_status({
                    "status": "running",
                    "attempt": attempt + 1,
                    "provider": PROVIDER,
                    "fallback": False,
                    "detail": f"повтор через {delay:.1f} с",
                })
                _wait(delay, deadline, is_cancelled)

        fallback_available = (
            error is not None
            and error.kind != "budget"
            and FALLBACK != "none"
            and FALLBACK != PROVIDER
            and not (PROVIDER == "openai" and FALLBACK == MODEL)
        )
        if fallback_available and not emitted:
            if is_cancelled():
                raise ProviderError("cancelled", "Генерацію скасовано.", attempts=attempts)
            provider_name = FALLBACK
            fallback_attempt = attempts + 1
            logger.warning("llm.fallback", extra={
                **common,
                "attempt": fallback_attempt,
                "provider": FALLBACK,
                "reason": error.kind,
            })
            try:
                return finish_result(run_provider(FALLBACK, fallback_attempt, True), True)
            except ProviderError as fallback_error:
                if fallback_error.kind == "cancelled" or is_cancelled():
                    raise ProviderError("cancelled", "Генерацію скасовано.", attempts=fallback_attempt) from fallback_error
                error = fallback_error
                if fallback_error.cost is not None:
                    save_failure(fallback_error)
                attempts = fallback_attempt

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
        previous_costs = [failure["cost"] for failure in failures]
        known_costs = [amount for amount in previous_costs if amount is not None]
        error.cost = round(sum(known_costs), 12) if len(known_costs) == len(previous_costs) else None
        error.cost_estimated = any(failure["cost_estimated"] for failure in failures)
        error.usage = _usage_totals(failures)
        error.attempts = attempts
        error.provider = provider_name
        raise error
    except ProviderError:
        raise
    except Exception as exc:
        logger.error("gateway.internal_error", extra={
            **common,
            "provider": provider_name,
            "error_type": type(exc).__name__,
        })
        raise ProviderError(
            "unknown", "Не вдалося створити чернетку.", attempts=attempts,
            provider=provider_name, detail=type(exc).__name__,
        ) from exc


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
