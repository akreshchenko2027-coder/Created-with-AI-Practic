import random
import os
import time
from typing import Iterator

from dotenv import load_dotenv

from .base import Chunk, ProviderError, Request, Result
from .. import samples

load_dotenv()

FIRST_DELAY_MS = int(os.getenv("MOCK_FIRST_DELAY_MS", "600"))
DELAY_MS = int(os.getenv("MOCK_DELAY_MS", "40"))
FAILURE_RATE = float(os.getenv("MOCK_FAILURE_RATE", "0"))
FAILURE_KIND = os.getenv("MOCK_FAILURE_KIND", "unavailable")
RETRY_AFTER = float(os.getenv("MOCK_RETRY_AFTER", "1"))

if FIRST_DELAY_MS < 0 or DELAY_MS < 0 or RETRY_AFTER < 0:
    raise ValueError("Затримки та Retry-After для mock не можуть бути відʼємними.")
if not 0 <= FAILURE_RATE <= 1:
    raise ValueError("MOCK_FAILURE_RATE має бути в межах від 0 до 1.")


class MockProvider:
    name = "mock"
    model = "mock-1"

    def complete(self, request: Request) -> Result:
        started = time.monotonic()
        chunks = list(self.stream(request))
        text = "".join(chunk.text for chunk in chunks)
        usage = next((chunk.usage for chunk in reversed(chunks) if chunk.usage), None)
        finish_reason = next((chunk.finish_reason for chunk in reversed(chunks) if chunk.finish_reason), None)
        return Result(text, self.name, self.model, usage, finish_reason, time.monotonic() - started)

    def stream(self, request: Request) -> Iterator[Chunk]:
        sample = samples.get(str(request.metadata.get("request_id", "")))
        if sample is None:
            raise ProviderError("bad_request", "Не вдалося знайти текст для чернетки.")
        text = str(sample.get("reply_example") or "").strip()
        if not text:
            raise ProviderError("bad_request", "Для звернення немає прикладу відповіді.")

        failure_kind = str(request.metadata.get("mock_failure_kind", FAILURE_KIND))
        failure_rate = float(request.metadata.get("mock_failure_rate", FAILURE_RATE))
        should_fail = random.random() < failure_rate
        cancelled = request.metadata.get("is_cancelled")
        attempt_timeout = float(request.metadata.get("attempt_timeout", 20))

        def wait(milliseconds: int) -> None:
            delay = max(0, milliseconds) / 1000
            if cancelled:
                deadline = time.monotonic() + delay
                while time.monotonic() < deadline:
                    if cancelled():
                        raise ProviderError("cancelled", "Генерацію скасовано.")
                    time.sleep(min(0.05, deadline - time.monotonic()))
            else:
                time.sleep(delay)

        first_delay = FIRST_DELAY_MS
        if should_fail and failure_kind == "timeout":
            first_delay = max(first_delay, int(attempt_timeout * 1000) + 100)
        wait(first_delay)
        if should_fail and failure_kind == "rate_limit":
            raise ProviderError(
                "rate_limit",
                "Провайдер тимчасово обмежив кількість запитів.",
                retryable=True,
                retry_after=float(request.metadata.get("mock_retry_after", RETRY_AFTER)),
            )
        if should_fail and failure_kind in {"unavailable", "auth", "bad_request"}:
            retryable = failure_kind == "unavailable"
            messages = {
                "unavailable": "Провайдер тимчасово недоступний.",
                "auth": "Не вдалося автентифікуватися у провайдера.",
                "bad_request": "Провайдер відхилив запит.",
            }
            raise ProviderError(failure_kind, messages[failure_kind], retryable=retryable)

        words = text.split(" ")
        emitted = []
        for index, word in enumerate(words):
            if index:
                wait(DELAY_MS)
            if should_fail and failure_kind == "mid_stream" and index == min(4, len(words) - 1):
                raise ProviderError(
                    "unavailable",
                    "Потік перервався після частини тексту.",
                    retryable=True,
                )
            piece = word if index == 0 else f" {word}"
            emitted.append(piece)
            yield Chunk(piece)
        result_text = "".join(emitted)
        completion_tokens = max(1, len(result_text.split()))
        prompt_tokens = max(1, sum(len(str(m.get("content", ""))) for m in request.messages) // 4)
        yield Chunk(
            "",
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            finish_reason="stop",
        )
