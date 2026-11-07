from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import time
from typing import Iterator

import httpx
import openai

from .base import Chunk, ProviderError, Request, Result


class OpenAICompatProvider:
    name = "openai"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.name = "openai"

    def _client(self, request: Request) -> openai.OpenAI:
        timeout = min(self.timeout, float(request.metadata.get("attempt_timeout", self.timeout)))
        idle = min(timeout, float(request.metadata.get("idle_timeout", timeout)))
        return openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=httpx.Timeout(timeout, read=idle),
            max_retries=0,
        )

    @staticmethod
    def _retry_after(exc: openai.APIStatusError) -> float | None:
        if exc.response is None:
            return None
        raw = exc.response.headers.get("retry-after")
        try:
            return max(0.0, float(raw)) if raw is not None else None
        except ValueError:
            try:
                date = parsedate_to_datetime(raw)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                return max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                return None

    @staticmethod
    def _provider_error(exc: Exception) -> ProviderError:
        if isinstance(exc, openai.APITimeoutError):
            return ProviderError("timeout", "Провайдер не відповів вчасно.", retryable=True)
        if isinstance(exc, openai.RateLimitError):
            return ProviderError(
                "rate_limit",
                "Провайдер тимчасово обмежив кількість запитів.",
                retryable=True,
                retry_after=OpenAICompatProvider._retry_after(exc),
                detail="RateLimitError",
            )
        if isinstance(exc, openai.AuthenticationError):
            return ProviderError("auth", "Не вдалося автентифікуватися у провайдера.", detail="AuthenticationError")
        if isinstance(exc, openai.BadRequestError):
            return ProviderError("bad_request", "Провайдер відхилив запит.", detail="BadRequestError")
        if isinstance(exc, openai.APIConnectionError):
            return ProviderError("unavailable", "Провайдер тимчасово недоступний.", retryable=True,
                                 detail="APIConnectionError")
        if isinstance(exc, openai.APIStatusError):
            if exc.status_code == 429:
                return ProviderError(
                    "rate_limit",
                    "Провайдер тимчасово обмежив кількість запитів.",
                    retryable=True,
                    retry_after=OpenAICompatProvider._retry_after(exc),
                    detail=f"HTTP {exc.status_code}",
                )
            retryable = exc.status_code >= 500
            kind = "unavailable" if retryable else "bad_request"
            message = "Провайдер тимчасово недоступний." if retryable else "Провайдер відхилив запит."
            return ProviderError(kind, message, retryable=retryable, detail=f"HTTP {exc.status_code}")
        if isinstance(exc, openai.OpenAIError):
            return ProviderError("unknown", "Не вдалося отримати відповідь від провайдера.",
                                 detail=type(exc).__name__)
        return ProviderError("unknown", "Не вдалося отримати відповідь від провайдера.",
                             detail=type(exc).__name__)

    def complete(self, request: Request) -> Result:
        started = time.monotonic()
        text = []
        usage = None
        finish_reason = None
        for chunk in self.stream(request):
            if chunk.text:
                text.append(chunk.text)
            if chunk.usage:
                usage = chunk.usage
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
        return Result(
            "".join(text),
            self.name,
            request.model or self.model,
            usage,
            finish_reason,
            time.monotonic() - started,
        )

    def stream(self, request: Request) -> Iterator[Chunk]:
        client = self._client(request)
        try:
            with client.chat.completions.create(
                model=request.model or self.model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            ) as response:
                for part in response:
                    choice = part.choices[0] if part.choices else None
                    content = choice.delta.content if choice and choice.delta else None
                    finish_reason = choice.finish_reason if choice else None
                    usage = part.usage.model_dump() if part.usage else None
                    yield Chunk(
                        content if isinstance(content, str) else "",
                        usage=usage,
                        finish_reason=finish_reason,
                    )
        except ProviderError:
            raise
        except Exception as exc:
            raise self._provider_error(exc) from exc
        finally:
            client.close()
