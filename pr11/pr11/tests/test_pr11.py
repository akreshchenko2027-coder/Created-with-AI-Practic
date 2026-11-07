import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import budget, cache, costs
from app.main import app
from app.providers.base import Request, Result


class CostTests(unittest.TestCase):
    def test_usage_cost_includes_cached_input_rate(self):
        amount, estimated = costs.estimate({
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 200},
        }, provider="openai")
        self.assertEqual(amount, 0.000505)
        self.assertFalse(estimated)

    def test_missing_usage_is_estimated(self):
        amount, estimated = costs.estimate(
            None, provider="openai", prompt="п" * 100, text="відповідь"
        )
        self.assertGreater(amount, 0)
        self.assertTrue(estimated)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.old = cache.CACHE_FILE, cache.TTL_SECONDS, cache.MAX_ENTRIES, cache.ENABLED
        cache.CACHE_FILE = str(Path(self.directory.name) / "cache.json")
        cache.TTL_SECONDS = 60
        cache.MAX_ENTRIES = 2
        cache.ENABLED = True
        cache._entries = None
        cache._counts.update(hits=0, misses=0, bypassed=0)

    def tearDown(self):
        cache.CACHE_FILE, cache.TTL_SECONDS, cache.MAX_ENTRIES, cache.ENABLED = self.old
        cache._entries = None
        self.directory.cleanup()

    def test_key_uses_prompt_version_and_generation_parameters(self):
        request = Request([{"role": "user", "content": "hello"}], 10, 0.2, "mock-1",
                          {"provider": "mock"})
        first = cache.make_key(request, "1")
        self.assertNotEqual(first, cache.make_key(request, "2"))
        request.temperature = 0.5
        self.assertNotEqual(first, cache.make_key(request, "1"))

    def test_persistence_and_expiration(self):
        request = Request([{"role": "user", "content": "hello"}], 10, 0.2, "mock-1",
                          {"provider": "mock"})
        key = cache.make_key(request, "1")
        result = Result("reply", "mock", "mock-1", {"prompt_tokens": 2, "completion_tokens": 1},
                        "stop", cost=0.00001)
        cache.put(key, result)
        cache._entries = None
        self.assertEqual(cache.get(key).text, "reply")
        cache._entries[key]["expires_at"] = time.time() - 1
        self.assertIsNone(cache.get(key))
        self.assertEqual(cache.stats()["hits"], 1)


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.old = budget.BUDGET_FILE, budget.DAILY_USD, budget.WARN_AT
        budget.BUDGET_FILE = str(Path(self.directory.name) / "budget.json")
        budget.DAILY_USD = 1.0
        budget.WARN_AT = 0.8

    def tearDown(self):
        budget.BUDGET_FILE, budget.DAILY_USD, budget.WARN_AT = self.old
        self.directory.cleanup()

    def test_concurrent_reservations_cannot_exceed_limit(self):
        start = threading.Barrier(2)
        results = []

        def reserve():
            start.wait()
            try:
                results.append(budget.check(0.6))
            except budget.BudgetExceeded:
                results.append(None)

        threads = [threading.Thread(target=reserve) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sum(result is not None for result in results), 1)
        state = budget.status()
        self.assertAlmostEqual(state["reserved"], 0.6)
        self.assertAlmostEqual(state["remaining"], 0.4)

    def test_record_persists_spend_and_releases_reservation(self):
        reservation = budget.check(0.5)
        budget.record(0.25, reservation)
        state = budget.status()
        self.assertEqual(state["spent"], 0.25)
        self.assertEqual(state["reserved"], 0)
        self.assertEqual(state["remaining"], 0.75)
        self.assertTrue(Path(budget.BUDGET_FILE).exists())


class ApiTests(unittest.TestCase):
    def test_correlation_cache_and_private_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            old_cache = cache.CACHE_FILE
            old_budget = budget.BUDGET_FILE
            cache.CACHE_FILE = str(Path(directory) / "cache.json")
            cache._entries = None
            cache._counts.update(hits=0, misses=0, bypassed=0)
            budget.BUDGET_FILE = str(Path(directory) / "budget.json")
            from app import telemetry

            log_path = Path(telemetry.LOG_FILE)
            offset = log_path.stat().st_size if log_path.exists() else 0
            try:
                with patch("app.gateway.PROVIDER", "mock"), patch("app.gateway.FALLBACK", "none"), \
                        TestClient(app) as client:
                    first = client.post("/api/drafts", json={"request_id": "d01", "client": "test-client"})
                    self.assertEqual(first.status_code, 202)
                    request_id = first.headers["X-Request-ID"]
                    self.assertEqual(first.json()["http_request_id"], request_id)
                    first_id = first.json()["draft_id"]
                    for _ in range(100):
                        snapshot = client.get(f"/api/drafts/{first_id}").json()
                        if snapshot["status"] in {"done", "failed"}:
                            break
                        time.sleep(0.02)
                    self.assertEqual(snapshot["status"], "done")

                    second = client.post("/api/drafts", json={"request_id": "d01", "client": "test-client"})
                    second_id = second.json()["draft_id"]
                    for _ in range(100):
                        cached = client.get(f"/api/drafts/{second_id}").json()
                        if cached["status"] in {"done", "failed"}:
                            break
                        time.sleep(0.02)
                    self.assertTrue(cached["cached"])
                    self.assertEqual(cached["cost"], 0.0)
                    self.assertGreater(cached["cache_saved_cost"], 0)

                    log_text = log_path.read_bytes()[offset:].decode("utf-8")
                    records = [json.loads(line) for line in log_text.splitlines() if line]
                    attempt = next(r for r in records if r.get("event") == "llm.attempt")
                    self.assertEqual(attempt["request_id"], request_id)
                    self.assertEqual(attempt["draft_id"], first_id)
                    serialized = log_text
                    self.assertNotIn("reply_example", serialized)
                    self.assertNotIn("Поверни лише текст листа", serialized)
            finally:
                cache.CACHE_FILE = old_cache
                cache._entries = None
                budget.BUDGET_FILE = old_budget


if __name__ == "__main__":
    unittest.main()
