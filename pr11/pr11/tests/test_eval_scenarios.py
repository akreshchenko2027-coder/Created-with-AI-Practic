import logging
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import budget, cache, drafting, gateway, samples
from app.main import app
from app.providers.base import ProviderError


class ScenarioTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.old_gateway = (
            gateway.PROVIDER,
            gateway.FALLBACK,
            gateway.MAX_ATTEMPTS,
            gateway.BACKOFF_BASE,
            gateway.BACKOFF_MAX,
            gateway.TOTAL_TIMEOUT,
            gateway.MAX_CONCURRENT,
            gateway._slots,
        )
        self.old_cache = cache.CACHE_FILE, cache.TTL_SECONDS, cache.MAX_ENTRIES, cache.ENABLED
        self.old_budget = budget.BUDGET_FILE, budget.DAILY_USD, budget.WARN_AT
        self.old_log_level = logging.getLogger("app.gateway").level
        self.budget_log = logging.getLogger("app.budget")
        self.old_budget_log_level = self.budget_log.level
        gateway.PROVIDER = "mock"
        gateway.FALLBACK = "none"
        gateway.MAX_ATTEMPTS = 3
        gateway.BACKOFF_BASE = 0.001
        gateway.BACKOFF_MAX = 0.001
        gateway.TOTAL_TIMEOUT = 30
        gateway.MAX_CONCURRENT = 5
        gateway._slots = threading.BoundedSemaphore(5)
        cache.CACHE_FILE = str(Path(self.directory.name) / "cache.json")
        cache.TTL_SECONDS = 60
        cache.MAX_ENTRIES = 100
        cache.ENABLED = True
        cache._entries = None
        cache._counts.update(hits=0, misses=0, bypassed=0)
        budget.BUDGET_FILE = str(Path(self.directory.name) / "budget.json")
        budget.DAILY_USD = 1.0
        budget.WARN_AT = 0.8
        self.mock_module = None
        from app.providers import mock

        self.mock_module = mock
        self.old_mock_settings = mock.FIRST_DELAY_MS, mock.DELAY_MS, mock.FAILURE_RATE, mock.FAILURE_KIND
        mock.FIRST_DELAY_MS = 0
        mock.DELAY_MS = 0
        mock.FAILURE_RATE = 0
        mock.FAILURE_KIND = "unavailable"
        self.handler = _RecordHandler()
        self.log = logging.getLogger("app.gateway")
        self.log.setLevel(logging.INFO)
        self.log.addHandler(self.handler)
        self.budget_log.setLevel(logging.INFO)
        self.budget_log.addHandler(self.handler)

    def tearDown(self):
        self.log.removeHandler(self.handler)
        self.budget_log.removeHandler(self.handler)
        gateway.PROVIDER, gateway.FALLBACK, gateway.MAX_ATTEMPTS, gateway.BACKOFF_BASE, \
            gateway.BACKOFF_MAX, gateway.TOTAL_TIMEOUT, gateway.MAX_CONCURRENT, gateway._slots = self.old_gateway
        self.log.setLevel(self.old_log_level)
        self.budget_log.setLevel(self.old_budget_log_level)
        cache.CACHE_FILE, cache.TTL_SECONDS, cache.MAX_ENTRIES, cache.ENABLED = self.old_cache
        cache._entries = None
        budget.BUDGET_FILE, budget.DAILY_USD, budget.WARN_AT = self.old_budget
        self.mock_module.FIRST_DELAY_MS, self.mock_module.DELAY_MS, \
            self.mock_module.FAILURE_RATE, self.mock_module.FAILURE_KIND = self.old_mock_settings
        self.directory.cleanup()

    def make_request(self, draft_id="scenario", fresh=False):
        request = drafting.build_request(samples.get("d01"))
        request.metadata.update({
            "http_request_id": f"http-{draft_id}",
            "draft_id": draft_id,
            "client": "scenario-test",
            "fresh": fresh,
            "provider": "mock",
        })
        return request

    def generate(self, request):
        return gateway.generate(request, lambda _: None, lambda _: None, lambda: False)

    def test_retry_trace_accounts_each_attempt_without_logging_private_text(self):
        self.mock_module.FAILURE_RATE = 0.6
        with patch("app.providers.mock.random.random", side_effect=[0.1, 0.1, 0.9]):
            result = self.generate(self.make_request("retry"))
        attempts = [record for record in self.handler.records if record.getMessage() == "llm.attempt"]
        retries = [record for record in self.handler.records if record.getMessage() == "llm.retry"]
        self.assertEqual(len(attempts), 3)
        self.assertEqual([record.attempt for record in attempts], [1, 2, 3])
        self.assertEqual([record.kind for record in attempts[:2]], ["unavailable", "unavailable"])
        self.assertEqual(len(retries), 2)
        self.assertGreater(result.cost, 0)
        self.assertTrue(result.cost_estimated)
        serialized = "\n".join(record.getMessage() + str(record.__dict__) for record in self.handler.records)
        self.assertNotIn(samples.get("d01")["text"], serialized)
        self.assertNotIn(samples.get("d01")["customer"], serialized)

    def test_five_simultaneous_identical_requests_are_not_coalesced(self):
        self.mock_module.FIRST_DELAY_MS = 100
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(
                lambda index: self.generate(self.make_request(f"parallel-{index}")),
                range(5),
            ))
        self.assertTrue(all(not result.cached for result in results))
        attempts = [record for record in self.handler.records if record.getMessage() == "llm.attempt"]
        self.assertEqual(len(attempts), 5)
        cached = self.generate(self.make_request("after-parallel"))
        self.assertTrue(cached.cached)
        self.assertEqual(cache.stats()["hits"], 1)

    def test_fresh_bypasses_cache_and_next_request_reuses_saved_result(self):
        first = self.generate(self.make_request("first"))
        repeated = self.generate(self.make_request("fresh", fresh=True))
        cached = self.generate(self.make_request("cached"))
        self.assertFalse(first.cached)
        self.assertFalse(repeated.cached)
        self.assertTrue(cached.cached)
        self.assertEqual(cache.stats()["bypassed"], 1)
        self.assertEqual(cache.stats()["hits"], 1)

    def test_prompt_version_change_misses_existing_cache(self):
        old_version = drafting.PROMPT_VERSION
        try:
            drafting.PROMPT_VERSION = "scenario-v1"
            first = self.generate(self.make_request("version-one"))
            drafting.PROMPT_VERSION = "scenario-v2"
            second = self.generate(self.make_request("version-two"))
        finally:
            drafting.PROMPT_VERSION = old_version
        self.assertFalse(first.cached)
        self.assertFalse(second.cached)
        attempts = [record for record in self.handler.records if record.getMessage() == "llm.attempt"]
        self.assertEqual(len(attempts), 2)

    def test_partial_stream_failure_is_costed_as_an_estimate(self):
        self.mock_module.FAILURE_RATE = 1
        self.mock_module.FAILURE_KIND = "mid_stream"
        gateway.MAX_ATTEMPTS = 1
        with self.assertRaises(ProviderError) as raised:
            self.generate(self.make_request("partial"))
        self.assertEqual(raised.exception.kind, "unavailable")
        self.assertGreater(raised.exception.cost, 0)
        self.assertTrue(raised.exception.cost_estimated)
        self.assertGreater(budget.status()["spent"], 0)
        self.assertTrue(any(
            record.getMessage() == "llm.attempt"
            and record.status == "failed"
            and record.cost_estimated
            for record in self.handler.records
        ))

    def test_auth_failure_does_not_retry(self):
        self.mock_module.FAILURE_RATE = 1
        self.mock_module.FAILURE_KIND = "auth"
        with self.assertRaises(ProviderError) as raised:
            self.generate(self.make_request("auth"))
        self.assertEqual(raised.exception.kind, "auth")
        attempts = [record for record in self.handler.records if record.getMessage() == "llm.attempt"]
        self.assertEqual(len(attempts), 1)

    def test_queue_time_is_separate_from_provider_time(self):
        gateway.MAX_CONCURRENT = 1
        gateway._slots = threading.BoundedSemaphore(1)
        self.mock_module.FIRST_DELAY_MS = 80
        start = threading.Barrier(3)

        def run(index):
            start.wait()
            return self.generate(self.make_request(f"queued-{index}"))

        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(run, range(3)))
        self.assertTrue(all(not result.cached for result in results))
        attempts = [record for record in self.handler.records if record.getMessage() == "llm.attempt"]
        self.assertEqual(len(attempts), 3)
        self.assertGreater(max(record.queue_ms for record in attempts), 0)
        self.assertTrue(all(record.duration_ms >= 70 for record in attempts))

    def test_slow_provider_delay_is_measured(self):
        self.mock_module.FIRST_DELAY_MS = 80
        started = time.monotonic()
        self.generate(self.make_request("slow"))
        elapsed = time.monotonic() - started
        attempt = next(
            record for record in self.handler.records if record.getMessage() == "llm.attempt"
        )
        self.assertGreaterEqual(elapsed, 0.07)
        self.assertGreaterEqual(attempt.first_token_ms, 70)

    def test_budget_rejection_is_visible_as_a_controlled_draft_error(self):
        budget.DAILY_USD = 0.00001
        with TestClient(app) as client:
            response = client.post("/api/drafts", json={"request_id": "d01", "client": "budget-test"})
            self.assertEqual(response.status_code, 202)
            self.assertIn("X-Request-ID", response.headers)
            draft_id = response.json()["draft_id"]
            snapshot = response.json()
            for _ in range(100):
                snapshot = client.get(f"/api/drafts/{draft_id}").json()
                if snapshot["status"] in {"done", "failed"}:
                    break
                time.sleep(0.01)
            usage = client.get("/api/usage").json()
        self.assertEqual(snapshot["status"], "failed")
        self.assertEqual(snapshot["error"]["kind"], "budget")
        self.assertIn("resets_at", usage["budget"])
        self.assertFalse(any(
            record.getMessage() == "llm.attempt" and getattr(record, "status", None) == "ok"
            for record in self.handler.records
        ))

    def test_budget_warning_rejection_and_restart_persistence(self):
        budget.DAILY_USD = 0.01
        reservation = budget.check(0.009, request_id="request", draft_id="draft")
        self.assertTrue(budget.status()["warning"])
        budget.record(0.0001, reservation)
        self.assertTrue(budget.status()["warning"])
        with TestClient(app) as client:
            usage = client.get("/api/usage").json()
        self.assertTrue(usage["budget"]["warning"])
        with self.assertRaises(budget.BudgetExceeded):
            budget.check(0.02)
        state = budget.status()
        self.assertAlmostEqual(state["spent"], 0.0001)
        self.assertTrue(state["warning"])
        self.assertTrue(any(
            record.getMessage() == "budget.warning"
            for record in self.handler.records
        ))


class _RecordHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


if __name__ == "__main__":
    unittest.main()
