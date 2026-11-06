import threading
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import drafting, gateway, jobs, limits, main, samples
from app.providers.base import Chunk, ProviderError, Result
from app.providers.mock import MockProvider


class ServiceTests(unittest.TestCase):
    def setUp(self):
        with jobs._changed:
            jobs._drafts.clear()
            jobs._order.clear()
            jobs._cancellations.clear()
        with limits._lock:
            limits._requests.clear()
        self.sample = samples.get("d01")

    def wait_for_terminal(self, draft, timeout=3):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            current = jobs.get(draft.draft_id)
            if current.status in jobs.TERMINAL:
                return current
            time.sleep(0.01)
        self.fail(f"Draft did not finish: {draft.status}")

    def mock_settings(self, **overrides):
        defaults = {
            "PROVIDER": "mock",
            "FALLBACK": "none",
            "MAX_ATTEMPTS": 2,
            "TIMEOUT": 1.0,
            "IDLE_TIMEOUT": 0.5,
            "TOTAL_TIMEOUT": 2.0,
            "BACKOFF_BASE": 0.01,
            "BACKOFF_MAX": 0.02,
        }
        defaults.update(overrides)
        return patch.multiple(gateway, **defaults)

    def test_request_contains_only_established_facts(self):
        request = drafting.build_request(self.sample)
        self.assertEqual(request.metadata["request_id"], "d01")
        self.assertIn(self.sample["text"], request.messages[1]["content"])
        self.assertIn(self.sample["facts"][0], request.messages[1]["content"])
        self.assertNotIn("reply_example", request.messages[1]["content"])

    def test_mock_provider_streams_sample_without_api_key(self):
        request = drafting.build_request(self.sample)
        with patch("app.providers.mock.FIRST_DELAY_MS", 1), patch("app.providers.mock.DELAY_MS", 0):
            chunks = list(MockProvider().stream(request))
        self.assertEqual("".join(chunk.text for chunk in chunks), self.sample["reply_example"])
        self.assertTrue(any(chunk.usage for chunk in chunks))

    def test_rate_limit_reports_retry_after(self):
        with patch.object(limits, "PER_MINUTE", 1):
            limits.acquire("client-a")
            with self.assertRaises(limits.RateLimited) as raised:
                limits.acquire("client-a")
        self.assertGreaterEqual(raised.exception.retry_after, 1)

    def test_rate_limit_recovers_after_window(self):
        with patch.object(limits, "PER_MINUTE", 1), patch.object(limits, "WINDOW", 0.03):
            limits.acquire("client-a")
            with self.assertRaises(limits.RateLimited):
                limits.acquire("client-a")
            time.sleep(0.04)
            limits.acquire("client-a")

    def test_gateway_retries_temporary_failure(self):
        class FlakyProvider:
            name = "mock"
            model = "test"

            def __init__(self):
                self.calls = 0

            def stream(self, request):
                self.calls += 1
                if self.calls == 1:
                    raise ProviderError("unavailable", "temporary", retryable=True)
                yield Chunk("ok")
                yield Chunk("", {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}, "stop")

            def complete(self, request):
                raise AssertionError

        provider = FlakyProvider()
        statuses = []
        deltas = []
        with self.mock_settings(), patch.object(gateway, "make_provider", return_value=provider):
            result = gateway.generate(drafting.build_request(self.sample), deltas.append, statuses.append, lambda: False)
        self.assertEqual(provider.calls, 2)
        self.assertEqual(result.text, "ok")
        self.assertEqual(deltas, ["ok"])
        self.assertEqual([item["attempt"] for item in statuses if item.get("provider")], [1, 2, 2])

    def test_persistent_mock_failure_stops_at_attempt_limit(self):
        factory = patch.object(gateway, "make_provider", wraps=gateway.make_provider)
        with self.mock_settings(MAX_ATTEMPTS=2), patch("app.providers.mock.FIRST_DELAY_MS", 0), patch(
            "app.providers.mock.FAILURE_RATE", 1
        ), patch("app.providers.mock.FAILURE_KIND", "unavailable"), factory as make_provider:
            with self.assertRaises(ProviderError) as raised:
                gateway.generate(drafting.build_request(self.sample), lambda _: None, lambda _: None, lambda: False)
        self.assertEqual(make_provider.call_count, 2)
        self.assertEqual(raised.exception.kind, "unavailable")

    def test_auth_failure_is_not_retried(self):
        class AuthProvider:
            name = "mock"
            model = "test"
            calls = 0

            def stream(self, request):
                self.calls += 1
                raise ProviderError("auth", "unauthorized")
                yield

            def complete(self, request):
                raise AssertionError

        provider = AuthProvider()
        with self.mock_settings(), patch.object(gateway, "make_provider", return_value=provider):
            with self.assertRaises(ProviderError) as raised:
                gateway.generate(drafting.build_request(self.sample), lambda _: None, lambda _: None, lambda: False)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(raised.exception.kind, "auth")

    def test_attempt_timeout_is_bounded_by_max_attempts(self):
        class SlowProvider:
            name = "mock"
            model = "test"
            calls = 0

            def stream(self, request):
                self.calls += 1
                time.sleep(0.08)
                yield Chunk("late")

            def complete(self, request):
                raise AssertionError

        provider = SlowProvider()
        with self.mock_settings(MAX_ATTEMPTS=2, TIMEOUT=0.03, IDLE_TIMEOUT=0.5, TOTAL_TIMEOUT=0.5), patch.object(
            gateway, "make_provider", return_value=provider
        ):
            with self.assertRaises(ProviderError) as raised:
                gateway.generate(drafting.build_request(self.sample), lambda _: None, lambda _: None, lambda: False)
        self.assertEqual(provider.calls, 2)
        self.assertEqual(raised.exception.kind, "timeout")

    def test_retry_after_over_budget_does_not_retry_or_wait(self):
        class LimitedProvider:
            name = "mock"
            model = "test"
            calls = 0

            def stream(self, request):
                self.calls += 1
                raise ProviderError("rate_limit", "limited", retryable=True, retry_after=1)
                yield

            def complete(self, request):
                raise AssertionError

        provider = LimitedProvider()
        started = time.monotonic()
        with self.mock_settings(MAX_ATTEMPTS=3, TOTAL_TIMEOUT=0.1), patch.object(
            gateway, "make_provider", return_value=provider
        ):
            with self.assertRaises(ProviderError) as raised:
                gateway.generate(drafting.build_request(self.sample), lambda _: None, lambda _: None, lambda: False)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(raised.exception.kind, "rate_limit")
        self.assertLess(time.monotonic() - started, 0.5)

    def test_retry_after_within_budget_is_respected(self):
        class LimitedOnceProvider:
            name = "mock"
            model = "test"

            def __init__(self):
                self.calls = 0

            def stream(self, request):
                self.calls += 1
                if self.calls == 1:
                    raise ProviderError("rate_limit", "limited", retryable=True, retry_after=0.03)
                yield Chunk("ok")

            def complete(self, request):
                raise AssertionError

        provider = LimitedOnceProvider()
        started = time.monotonic()
        with self.mock_settings(TOTAL_TIMEOUT=0.5), patch.object(gateway, "make_provider", return_value=provider):
            result = gateway.generate(drafting.build_request(self.sample), lambda _: None, lambda _: None, lambda: False)
        self.assertEqual(result.text, "ok")
        self.assertEqual(provider.calls, 2)
        self.assertGreaterEqual(time.monotonic() - started, 0.025)

    def test_total_timeout_stops_before_all_attempts(self):
        class SlowProvider:
            name = "mock"
            model = "test"
            calls = 0

            def stream(self, request):
                self.calls += 1
                time.sleep(0.08)
                yield Chunk("late")

            def complete(self, request):
                raise AssertionError

        provider = SlowProvider()
        with self.mock_settings(MAX_ATTEMPTS=3, TIMEOUT=0.5, IDLE_TIMEOUT=0.5, TOTAL_TIMEOUT=0.03), patch.object(
            gateway, "make_provider", return_value=provider
        ):
            with self.assertRaises(ProviderError) as raised:
                gateway.generate(drafting.build_request(self.sample), lambda _: None, lambda _: None, lambda: False)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(raised.exception.kind, "timeout")

    def test_fallback_is_reported_and_uses_its_model(self):
        class Provider:
            model = "fallback-model"

            def __init__(self, name, fails):
                self.name = name
                self.fails = fails

            def stream(self, request):
                if self.fails:
                    raise ProviderError("auth", "unauthorized")
                self.assert_fallback_model(request.model)
                yield Chunk("fallback")
                yield Chunk("", finish_reason="stop")

            def assert_fallback_model(self, model):
                if model != "fallback-model":
                    raise AssertionError(f"Wrong fallback model: {model}")

            def complete(self, request):
                raise AssertionError

        primary = Provider("openai", True)
        fallback = Provider("fallback-model", False)
        statuses = []

        def make_provider(name):
            return primary if name == "openai" else fallback

        with self.mock_settings(PROVIDER="openai", FALLBACK="fallback-model"), patch.object(
            gateway, "make_provider", side_effect=make_provider
        ):
            result = gateway.generate(drafting.build_request(self.sample), lambda _: None, statuses.append, lambda: False)
        self.assertEqual(result.text, "fallback")
        self.assertTrue(any(item.get("fallback") for item in statuses))

    def test_partial_stream_failure_is_not_retried(self):
        class PartialProvider:
            name = "mock"
            model = "test"
            calls = 0

            def stream(self, request):
                self.calls += 1
                yield Chunk("partial")
                raise ProviderError("unavailable", "interrupted", retryable=True)

            def complete(self, request):
                raise AssertionError

        provider = PartialProvider()
        deltas = []
        with self.mock_settings(), patch.object(gateway, "make_provider", return_value=provider):
            with self.assertRaises(ProviderError) as raised:
                gateway.generate(drafting.build_request(self.sample), deltas.append, lambda _: None, lambda: False)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(deltas, ["partial"])
        self.assertTrue(raised.exception.retryable)
        self.assertIn("частини тексту", raised.exception.message)

    def test_idle_timeout_stops_stream(self):
        class QuietProvider:
            name = "mock"
            model = "test"

            def stream(self, request):
                time.sleep(0.2)
                yield Chunk("late")

            def complete(self, request):
                raise AssertionError

        with self.mock_settings(MAX_ATTEMPTS=1, IDLE_TIMEOUT=0.03), patch.object(
            gateway, "make_provider", return_value=QuietProvider()
        ):
            with self.assertRaises(ProviderError) as raised:
                gateway.generate(drafting.build_request(self.sample), lambda _: None, lambda _: None, lambda: False)
        self.assertEqual(raised.exception.kind, "timeout")

    def test_background_job_streams_and_finishes(self):
        with self.mock_settings(), patch("app.providers.mock.FIRST_DELAY_MS", 1), patch(
            "app.providers.mock.DELAY_MS", 0
        ):
            draft = jobs.start(self.sample, "client")
            events = list(jobs.subscribe(draft.draft_id, heartbeat=0.02))
        result = self.wait_for_terminal(draft)
        self.assertEqual(result.status, "done")
        self.assertTrue(result.text)
        self.assertIn("delta", [item["event"] for item in events])
        self.assertEqual(events[-1]["event"], "done")

    def test_cancel_prevents_completion_and_preserves_partial_text(self):
        with self.mock_settings(), patch("app.providers.mock.FIRST_DELAY_MS", 1), patch(
            "app.providers.mock.DELAY_MS", 100
        ):
            draft = jobs.start(self.sample, "client")
            deadline = time.monotonic() + 1
            while draft.status == "queued" and time.monotonic() < deadline:
                time.sleep(0.005)
            time.sleep(0.02)
            cancelled = jobs.cancel(draft.draft_id)
            time.sleep(0.1)
        self.assertEqual(cancelled.status, "cancelled")
        self.assertNotIn("done", [item["event"] for item in cancelled.events])

    def test_cancel_missing_and_finished_drafts(self):
        with self.assertRaises(jobs.DraftNotFound):
            jobs.cancel("missing")
        draft = jobs.new_draft(self.sample["id"], "client")
        with jobs._changed:
            draft.status = "done"
        with self.assertRaises(jobs.DraftAlreadyFinished):
            jobs.cancel(draft.draft_id)

    def test_http_errors_and_retry_after(self):
        client = TestClient(main.app)
        with patch.object(limits, "PER_MINUTE", 1), self.mock_settings():
            first = client.post("/api/drafts", json={"request_id": "d01", "client": "client"})
            second = client.post("/api/drafts", json={"request_id": "d01", "client": "client"})
        unknown = client.post("/api/drafts", json={"request_id": "unknown", "client": "other"})
        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 429)
        self.assertTrue(second.headers.get("Retry-After"))
        self.assertEqual(second.json()["error"]["kind"], "rate_limit")
        self.assertEqual(unknown.status_code, 404)

    def test_api_cancel_finished_and_unknown(self):
        client = TestClient(main.app)
        self.assertEqual(client.post("/api/drafts/missing/cancel").status_code, 404)
        draft = jobs.new_draft(self.sample["id"], "client")
        with jobs._changed:
            draft.status = "done"
        response = client.post(f"/api/drafts/{draft.draft_id}/cancel")
        self.assertEqual(response.status_code, 409)


if __name__ == "__main__":
    unittest.main()
