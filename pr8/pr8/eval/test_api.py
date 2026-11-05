import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import assistant
from app.main import app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_blank_question_is_rejected(self):
        response = self.client.post(
            "/api/ask",
            json={"customer_id": "C-1001", "question": "   "},
        )
        self.assertEqual(response.status_code, 422)

    def test_unknown_customer_is_rejected_before_model_call(self):
        with patch("app.main.assistant.answer") as answer:
            response = self.client.post(
                "/api/ask",
                json={"customer_id": "C-9999", "question": "Де моє замовлення?"},
            )
        self.assertEqual(response.status_code, 404)
        answer.assert_not_called()

    def test_model_error_is_returned_as_gateway_error(self):
        with patch(
            "app.main.assistant.answer",
            side_effect=assistant.llm.LLMError("Модель недоступна.", 503),
        ):
            response = self.client.post(
                "/api/ask",
                json={"customer_id": "C-1001", "question": "Де моє замовлення?"},
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Модель недоступна.")

    def test_success_response_matches_page_contract(self):
        result = assistant.Answer(text="Готово.", rounds=1, elapsed={"model": 0.1, "tools": 0.0})
        with patch("app.main.assistant.answer", return_value=result):
            response = self.client.post(
                "/api/ask",
                json={"customer_id": "C-1001", "question": "Дякую"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Готово.")
        self.assertEqual(response.json()["rounds"], 1)


if __name__ == "__main__":
    unittest.main()
