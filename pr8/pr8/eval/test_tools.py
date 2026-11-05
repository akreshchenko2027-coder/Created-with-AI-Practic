import json
import unittest
from unittest.mock import patch

from app import tools
from shop import service


class ToolTests(unittest.TestCase):
    def setUp(self):
        service.reset()
        self.ctx = tools.Context(customer_id="C-1001")

    def test_specs_expose_only_client_operations(self):
        names = {item["function"]["name"] for item in tools.specs()}
        self.assertEqual(
            names,
            {
                "list_orders",
                "get_order",
                "search_products",
                "get_product",
                "get_stock",
                "delivery_quote",
                "create_return",
            },
        )
        self.assertFalse(any(name in names for name in ("issue_refund", "update_price", "add_bonus")))

    def test_owned_order_contains_only_client_fields(self):
        result = tools.call("get_order", '{"order_id":"10458"}', self.ctx)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.content["delivery"]["tracking"], "20450012345678")
        self.assertNotIn("customer_id", result.content)
        self.assertNotIn("internal_note", result.content)
        self.assertNotIn("payment", result.content)

    def test_foreign_order_is_rejected_without_returning_order_data(self):
        result = tools.call("get_order", '{"order_id":"10466"}', self.ctx)
        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.content["code"], "forbidden")
        self.assertNotIn("delivery", result.content)
        self.assertNotIn("10466", result.reason)

    def test_invalid_schema_does_not_call_service(self):
        with patch("app.tools.service.get_stock") as get_stock:
            result = tools.call(
                "get_stock",
                json.dumps({"sku": "ALT-AX3", "extra": "value"}),
                self.ctx,
            )
        self.assertEqual(result.status, "rejected")
        get_stock.assert_not_called()

    def test_zero_delivery_quantity_is_rejected_before_service(self):
        arguments = {
            "city": "Львів",
            "method": "branch",
            "items": [{"sku": "ALT-AX3", "quantity": 0}],
        }
        with patch("app.tools.service.delivery_quote") as delivery_quote:
            result = tools.call("delivery_quote", json.dumps(arguments), self.ctx)
        self.assertEqual(result.status, "rejected")
        delivery_quote.assert_not_called()

    def test_malformed_json_is_rejected(self):
        result = tools.call("get_stock", "{", self.ctx)
        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.content["code"], "invalid_json")

    def test_unknown_tool_is_rejected(self):
        result = tools.call("issue_refund", '{"order_id":"10458","amount":"1"}', self.ctx)
        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.content["code"], "unknown_tool")

    def test_foreign_return_is_rejected_before_mutation(self):
        with (
            patch("app.tools.service.get_order", return_value={"customer_id": "C-1002"}),
            patch("app.tools.service.create_return") as create_return,
        ):
            result = tools.call(
                "create_return",
                '{"order_id":"10466","sku":"SIR-R10","reason":"not_suitable","quantity":1}',
                self.ctx,
            )
        self.assertEqual(result.status, "rejected")
        create_return.assert_not_called()

    def test_valid_return_returns_only_customer_facing_fields(self):
        result = tools.call(
            "create_return",
            '{"order_id":"10412","sku":"SIR-R10","reason":"not_suitable","quantity":1,"comment":"Розмір"}',
            tools.Context(customer_id="C-1003"),
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.content["return_id"], "R-5001")
        self.assertNotIn("comment", result.content)

    def test_shop_unavailability_is_returned_as_tool_error(self):
        with patch("app.tools.service.get_stock", side_effect=service.Unavailable("offline")):
            result = tools.call("get_stock", '{"sku":"ALT-AX3"}', self.ctx)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.content["code"], "unavailable")


if __name__ == "__main__":
    unittest.main()
