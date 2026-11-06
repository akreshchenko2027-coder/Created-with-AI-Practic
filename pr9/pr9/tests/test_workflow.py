import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import main, schema, store, tools, workflow
from app.state import Run
from shop import service


def fake_complete(messages, schema=None, timeout=None, max_tokens=None):
    prompt = messages[-1]["content"]
    properties = schema["properties"]
    if "intent" in properties:
        request = re.search(r"<request>(.*?)</request>", prompt, re.DOTALL).group(1)
        lower = request.casefold()
        has_return = "поверн" in lower or "поверни" in lower
        has_cancel = "скасу" in lower
        has_status = "статус" in lower or "де моє" in lower or "де зараз" in lower
        if has_return and has_cancel:
            intent = "multiple"
        elif has_return:
            intent = "return"
        elif has_cancel:
            intent = "cancel"
        elif has_status:
            intent = "status"
        else:
            intent = "other"
        if "фільтр" in lower:
            item = "фільтри"
        elif "пилосос" in lower:
            item = "пилосос"
        elif "роутер" in lower:
            item = "роутер"
        elif "годинник" in lower:
            item = "годинник"
        elif "навушник" in lower:
            item = "навушники"
        else:
            item = None
        order_match = re.search(r"(?<!\d)(\d{5})(?!\d)", request)
        reason = None
        if any(word in lower for word in ("defect", "дефект", "заряд", "не працю")):
            reason = "defect"
        elif any(word in lower for word in ("не підійш", "не знадоб", "не потріб", "завелик")):
            reason = "not_suitable"
        elif "пошкод" in lower:
            reason = "damaged"
        elif "не той товар" in lower:
            reason = "wrong_item"
        quantity = 2 if "обидві" in lower else None
        answer = {
            "intent": intent,
            "order_id": order_match.group(1) if order_match else None,
            "item": item,
            "reason": reason,
            "quantity": quantity,
            "multiple_actions": has_return and has_cancel,
        }
    elif "sku" in properties:
        enum_values = properties["sku"]["enum"]
        answer = {"sku": next((sku for sku in enum_values if sku is not None), None)}
    else:
        facts_match = re.search(r"Факти: (.*)$", prompt, re.DOTALL)
        facts = json.loads(facts_match.group(1)) if facts_match else {}
        if "return" in facts:
            answer = {"reply": f"Заявку {facts['return']['return_id']} створено."}
        elif "order" in facts:
            order = facts["order"]
            if order["status"] == "Скасовано":
                reply = f"Замовлення {order['order_id']} скасовано."
            else:
                reply = f"Замовлення {order['order_id']}: {order['status']}."
            answer = {"reply": reply}
        elif "message" in facts:
            message = facts["message"]
            if "очікує підтвердження" in message:
                reply = message
            elif "змінився" in message:
                reply = f"Повернення не можна виконати. {message}"
            else:
                reply = message
            answer = {"reply": reply}
        else:
            answer = {"reply": "Звернення перевірено."}
    return {
        "text": json.dumps(answer, ensure_ascii=False),
        "model": "test",
        "elapsed": 0.001,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old_runs_dir = store.RUNS_DIR
        store.RUNS_DIR = Path(self.temp.name)
        service.reset()
        self.model = patch.object(workflow.llm, "complete", side_effect=fake_complete)
        self.model.start()

    def tearDown(self):
        self.model.stop()
        store.RUNS_DIR = self.old_runs_dir
        self.temp.cleanup()

    def test_return_requires_approval_and_is_idempotent(self):
        run = workflow.start(
            "Хочу повернути робот-пилосос із замовлення 10412, він не підійшов.",
            "C-1003",
        )
        self.assertEqual(run.status, "awaiting_approval")
        self.assertEqual(service.list_returns(), [])
        saved = store.load(run.run_id)
        self.assertEqual(saved.status, "awaiting_approval")

        completed = workflow.resume(run.run_id, True)
        self.assertEqual(completed.status, "done")
        self.assertEqual(len(service.list_returns()), 1)

        repeated = workflow.resume(run.run_id, True)
        self.assertEqual(repeated.status, "done")
        self.assertEqual(len(service.list_returns()), 1)
        self.assertEqual(repeated.steps[-1].name, "duplicate_decision")

    def test_duplicate_request_reuses_existing_return(self):
        first = workflow.start(
            "Хочу повернути робот-пилосос із замовлення 10412, він не підійшов.",
            "C-1003",
        )
        workflow.resume(first.run_id, True)
        duplicate = workflow.start(
            "Повторно хочу повернути пилосос із замовлення 10412, він не підійшов.",
            "C-1003",
        )
        self.assertEqual(duplicate.status, "done")
        self.assertIn("R-5001", duplicate.reply)
        self.assertEqual(len(service.list_returns()), 1)

    def test_duplicate_request_waiting_for_approval_is_not_queued_twice(self):
        first = workflow.start(
            "Хочу повернути робот-пилосос із замовлення 10412, він не підійшов.",
            "C-1003",
        )
        second = workflow.start(
            "Повторно хочу повернути пилосос із замовлення 10412, він не підійшов.",
            "C-1003",
        )
        self.assertEqual(first.status, "awaiting_approval")
        self.assertEqual(second.status, "done")
        self.assertIsNone(second.pending)
        self.assertEqual(len(store.list_runs("awaiting_approval")), 1)
        self.assertEqual(service.list_returns(), [])

    def test_unowned_order_does_not_expose_details(self):
        run = workflow.start(
            "Поверніть пилосос із замовлення 10466.",
            "C-1001",
        )
        self.assertEqual(run.status, "needs_info")
        self.assertNotIn("10466", run.reply)
        self.assertNotIn("data", run.data)
        self.assertNotIn("items", json.dumps(run.steps, default=lambda item: item.__dict__))

    def test_multiple_actions_are_handed_off_without_changes(self):
        run = workflow.start(
            "Скасуйте замовлення 10463 і поверніть товар із 10421.",
            "C-1001",
        )
        self.assertEqual(run.status, "handed_off")
        self.assertEqual(run.route, "multiple")
        self.assertEqual(service.list_returns(), [])
        self.assertEqual(service.get_order("10463")["status"], "preparing")

    def test_status_question_overrides_wrong_model_intent(self):
        real_complete = fake_complete

        def incorrect_classifier(messages, schema=None, **kwargs):
            if schema and "intent" in schema["properties"]:
                return {
                    "text": json.dumps(
                        {
                            "intent": "return",
                            "order_id": None,
                            "item": None,
                            "reason": None,
                            "quantity": None,
                            "multiple_actions": False,
                        }
                    ),
                    "model": "test",
                    "elapsed": 0.001,
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }
            return real_complete(
                messages,
                schema=schema,
                **kwargs,
            )

        with patch.object(workflow.llm, "complete", side_effect=incorrect_classifier):
            run = workflow.start("Де моє замовлення з годинником?", "C-1001")
        self.assertEqual(run.route, "status")
        self.assertEqual(run.status, "done")
        self.assertIn("10458", run.reply)

    def test_undelivered_return_is_rejected_without_reason_or_item_guess(self):
        run = workflow.start(
            "Хочу повернути годинник Вега S із замовлення 10458.",
            "C-1001",
        )
        self.assertEqual(run.route, "return")
        self.assertEqual(run.status, "done")
        self.assertIn("отримання замовлення", run.reply)
        self.assertEqual(service.list_returns(), [])

    def test_policy_is_checked_again_after_approval(self):
        run = workflow.start(
            "Хочу повернути робот-пилосос із замовлення 10412, він не підійшов.",
            "C-1003",
        )
        real_call = tools.call

        def changed_policy(name, raw_arguments, context):
            if name == "check_return":
                return tools.ToolResult(
                    status="ok",
                    content={
                        "eligible": False,
                        "policy_issue": "Статус замовлення змінився.",
                        "duplicate": None,
                    },
                )
            return real_call(name, raw_arguments, context)

        with patch.object(workflow.tools, "call", side_effect=changed_policy):
            completed = workflow.resume(run.run_id, True)
        self.assertEqual(completed.status, "done")
        self.assertEqual(service.list_returns(), [])
        self.assertIn("Статус замовлення змінився", completed.reply)

    def test_explicit_accessory_overrides_wrong_model_item(self):
        def wrong_item_classifier(messages, schema=None, **kwargs):
            result = fake_complete(messages, schema=schema, **kwargs)
            if schema and "intent" in schema["properties"]:
                parsed = json.loads(result["text"])
                parsed["item"] = "пилосос"
                result["text"] = json.dumps(parsed, ensure_ascii=False)
            return result

        with patch.object(workflow.llm, "complete", side_effect=wrong_item_classifier):
            run = workflow.start(
                "Фільтри до пилососа із замовлення 10412 не знадобились, хочу повернути обидві упаковки.",
                "C-1003",
            )

        self.assertEqual(run.data["parsed"]["item"], "фільтри")
        self.assertEqual(run.data["outcome"], "policy")
        self.assertEqual(service.list_returns(), [])
        self.assertFalse(any(step.name == "approval" for step in run.steps))

    def test_step_limit_stops_before_a_shop_mutation(self):
        with patch.object(workflow, "MAX_STEPS", 1):
            run = workflow.start(
                "Хочу повернути робот-пилосос із замовлення 10412, він не підійшов.",
                "C-1003",
            )
        self.assertEqual(run.status, "stopped")
        self.assertIn("крок", run.stopped)
        self.assertEqual(service.list_returns(), [])

    def test_token_and_time_limits_stop_before_a_shop_mutation(self):
        with patch.object(workflow, "TOKEN_BUDGET", 10):
            token_limited = workflow.start(
                "Хочу повернути робот-пилосос із замовлення 10412, він не підійшов.",
                "C-1003",
            )
        self.assertEqual(token_limited.status, "stopped")
        self.assertIn("токен", token_limited.stopped)
        self.assertEqual(service.list_returns(), [])

        with patch.object(workflow, "TIMEOUT", 1e-12):
            time_limited = workflow.start(
                "Хочу повернути робот-пилосос із замовлення 10412, він не підійшов.",
                "C-1003",
            )
        self.assertEqual(time_limited.status, "stopped")
        self.assertIn("час", time_limited.stopped)
        self.assertEqual(service.list_returns(), [])

    def test_startup_recovery_never_repeats_uncertain_mutation(self):
        run = Run(
            run_id="a" * 32,
            customer_id="C-1003",
            text="Повернення",
            status="executing",
            pending={
                "tool": "create_return",
                "arguments": {
                    "order_id": "10412",
                    "sku": "SIR-R10",
                    "reason": "not_suitable",
                    "quantity": 1,
                },
            },
            data={
                "mutation_intent": {
                    "tool": "create_return",
                    "arguments": {
                        "order_id": "10412",
                        "sku": "SIR-R10",
                        "reason": "not_suitable",
                        "quantity": 1,
                    },
                }
            },
            usage={"total_tokens": 0},
            elapsed={"total": 0.0},
        )
        store.save(run)
        workflow.recover_incomplete()
        recovered = store.load(run.run_id)
        self.assertEqual(recovered.status, "failed")
        self.assertIn("Повторного виклику не було", recovered.stopped)
        self.assertEqual(service.list_returns(), [])

    def test_approval_queue_survives_startup_recovery(self):
        run = workflow.start(
            "Хочу повернути робот-пилосос із замовлення 10412, він не підійшов.",
            "C-1003",
        )
        self.assertEqual(run.status, "awaiting_approval")

        workflow.recover_incomplete()
        recovered = store.load(run.run_id)
        self.assertEqual(recovered.status, "awaiting_approval")

        completed = workflow.resume(run.run_id, True)
        self.assertEqual(completed.status, "done")
        self.assertEqual(len(service.list_returns()), 1)

    def test_unverified_delivery_date_is_replaced_with_safe_status(self):
        real_complete = fake_complete

        def hallucinated_date(messages, schema=None, **kwargs):
            if schema and "reply" in schema["properties"]:
                return {
                    "text": json.dumps(
                        {"reply": "Замовлення 10458: Передано в доставку, доставка відбудеться 03.10.2026."},
                        ensure_ascii=False,
                    ),
                    "model": "test",
                    "elapsed": 0.001,
                    "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
                }
            return real_complete(messages, schema=schema, **kwargs)

        with patch.object(workflow.llm, "complete", side_effect=hallucinated_date):
            run = workflow.start("Де моє замовлення з годинником?", "C-1001")
        self.assertEqual(run.status, "done")
        self.assertNotIn("03.10.2026", run.reply)
        self.assertIn("10458", run.reply)
        self.assertIn("спрогнозовано", run.data["reply_validation"]["reason"])

    def test_schema_rejects_invented_or_extra_fields(self):
        valid = {
            "intent": "return",
            "order_id": None,
            "item": "пилосос",
            "reason": None,
            "quantity": None,
            "multiple_actions": False,
        }
        self.assertEqual(schema.validate(json.dumps(valid)), valid)
        with self.assertRaises(ValueError):
            schema.validate(json.dumps({**valid, "order_id": "104123"}))
        with self.assertRaises(ValueError):
            schema.validate(json.dumps({**valid, "approved": True}))

    def test_api_handles_duplicate_approval_and_unknown_run(self):
        response = main.api_create(
            main.RequestIn(
                customer_id="C-1003",
                text="Хочу повернути робот-пилосос із замовлення 10412, він не підійшов.",
            )
        )
        run_id = response["run_id"]
        first = main.api_decision(run_id, main.DecisionIn(approve=True))
        second = main.api_decision(run_id, main.DecisionIn(approve=True))
        self.assertEqual(first["status"], "done")
        self.assertEqual(second["status"], "done")
        self.assertEqual(len(service.list_returns()), 1)
        with self.assertRaises(main.HTTPException) as raised:
            main.api_run("0" * 32)
        self.assertEqual(raised.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
