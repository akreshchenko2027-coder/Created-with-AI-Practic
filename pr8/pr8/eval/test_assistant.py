import unittest
from unittest.mock import patch

from app import assistant, tools


def tool_call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class AssistantTests(unittest.TestCase):
    def setUp(self):
        self.first = {
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call("call-1", "get_order", '{"order_id":"10458"}')],
            },
            "elapsed": 0.2,
            "model": "test-model",
            "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
        }

    def test_tool_result_is_returned_to_model_before_final_answer(self):
        final = {
            "message": {"role": "assistant", "content": "Замовлення передано в доставку."},
            "elapsed": 0.1,
            "model": "test-model",
            "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
        }
        tool_result = tools.ToolResult(status="ok", content={"status": "Передано в доставку"})
        with (
            patch.object(assistant, "MAX_ROUNDS", 3),
            patch.object(assistant.llm, "chat", side_effect=[self.first, final]) as chat,
            patch.object(assistant.tools, "call", return_value=tool_result),
        ):
            result = assistant.answer("Де моє замовлення?", "C-1001")

        self.assertEqual(result.rounds, 2)
        self.assertEqual(result.stopped, "answer")
        self.assertEqual(result.text, "Замовлення передано в доставку.")
        self.assertEqual(result.usage["total_tokens"], 44)
        self.assertEqual(result.calls[0].status, "ok")
        self.assertTrue(any(message["role"] == "tool" for message in chat.call_args_list[1].args[0]))

    def test_round_limit_returns_an_explicit_incomplete_result(self):
        with (
            patch.object(assistant, "MAX_ROUNDS", 1),
            patch.object(assistant.llm, "chat", return_value=self.first),
            patch.object(
                assistant.tools,
                "call",
                return_value=tools.ToolResult(
                    status="error",
                    content={"error": "offline", "code": "unavailable"},
                ),
            ),
        ):
            result = assistant.answer("Де моє замовлення?", "C-1001")

        self.assertEqual(result.stopped, "limit")
        self.assertIn("тимчасово недоступний", result.text)

    def test_multiple_mutations_in_one_response_are_all_rejected(self):
        response = {
            **self.first,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    tool_call("call-1", "create_return", "{}"),
                    tool_call("call-2", "create_return", "{}"),
                ],
            },
        }
        with (
            patch.object(assistant, "MAX_ROUNDS", 1),
            patch.object(assistant.llm, "chat", return_value=response),
            patch.object(assistant.tools, "call") as call,
        ):
            result = assistant.answer("Оформіть повернення.", "C-1001")

        call.assert_not_called()
        self.assertEqual([item.status for item in result.calls], ["rejected", "rejected"])

    def test_return_is_not_created_twice_across_model_rounds(self):
        create = tool_call(
            "call-1",
            "create_return",
            '{"order_id":"10412","sku":"SIR-R10","reason":"not_suitable","quantity":1}',
        )
        first = {
            **self.first,
            "message": {"role": "assistant", "content": None, "tool_calls": [create]},
        }
        second = {
            **self.first,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call("call-2", "create_return", create["function"]["arguments"])],
            },
        }
        final = {
            "message": {"role": "assistant", "content": "Заявку вже оформлено."},
            "elapsed": 0.1,
            "model": "test-model",
            "usage": None,
        }
        with (
            patch.object(assistant, "MAX_ROUNDS", 3),
            patch.object(assistant.llm, "chat", side_effect=[first, second, final]),
            patch.object(
                assistant.tools,
                "call",
                return_value=tools.ToolResult(status="ok", content={"return_id": "R-5001"}),
            ) as call,
        ):
            result = assistant.answer("Оформіть повернення.", "C-1003")

        call.assert_called_once()
        self.assertEqual([item.status for item in result.calls], ["ok", "rejected"])
        self.assertEqual(result.calls[1].result["code"], "duplicate_mutation")


if __name__ == "__main__":
    unittest.main()
