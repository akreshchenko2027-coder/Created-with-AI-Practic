import copy
import json
import logging
import threading
from dataclasses import dataclass
from datetime import date, timedelta

from jsonschema import Draft202012Validator

from shop import service

logger = logging.getLogger(__name__)


@dataclass
class Context:
    customer_id: str


@dataclass
class ToolResult:
    status: str
    content: dict | list | str | None = None
    reason: str | None = None
    arguments: dict | None = None


def _object(properties: dict, required: list[str] | None = None) -> dict:
    schema = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


_TOOLS = [
    {
        "name": "list_orders",
        "description": "Показати замовлення поточного клієнта. Не приймає ідентифікатор клієнта.",
        "parameters": _object({}),
    },
    {
        "name": "get_order",
        "description": "Перевірити стан замовлення за його номером. Доступні лише замовлення поточного клієнта.",
        "parameters": _object(
            {"order_id": {"type": "string", "pattern": "^[0-9]{5}$"}},
            ["order_id"],
        ),
    },
    {
        "name": "search_products",
        "description": "Знайти товари за назвою або описом. Використовуйте перед запитом картки, наявності чи доставки, якщо артикул невідомий.",
        "parameters": _object(
            {
                "query": {"type": "string", "minLength": 1, "maxLength": 120},
                "category": {"type": "string", "minLength": 1, "maxLength": 80},
                "max_price": {"type": "number", "exclusiveMinimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            ["query"],
        ),
    },
    {
        "name": "get_product",
        "description": "Показати ціну, характеристики та опис товару за артикулом.",
        "parameters": _object(
            {"sku": {"type": "string", "pattern": "^[A-Z0-9-]{2,20}$"}},
            ["sku"],
        ),
    },
    {
        "name": "get_stock",
        "description": "Перевірити залишок товару на складі та очікувану дату поставки.",
        "parameters": _object(
            {"sku": {"type": "string", "pattern": "^[A-Z0-9-]{2,20}$"}},
            ["sku"],
        ),
    },
    {
        "name": "delivery_quote",
        "description": "Розрахувати вартість і строк доставки відомих товарів у вказане місто.",
        "parameters": _object(
            {
                "city": {"type": "string", "minLength": 1, "maxLength": 80},
                "method": {"type": "string", "enum": ["branch", "courier", "pickup"]},
                "items": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 10,
                    "items": _object(
                        {
                            "sku": {"type": "string", "pattern": "^[A-Z0-9-]{2,20}$"},
                            "quantity": {"type": "integer", "minimum": 1, "maximum": 50},
                        },
                        ["sku", "quantity"],
                    ),
                },
            },
            ["city", "method", "items"],
        ),
    },
    {
        "name": "create_return",
        "description": "Оформити заявку на повернення товару з отриманого замовлення поточного клієнта. Викликайте лише за його прямим проханням; причину й кількість беріть із повідомлення клієнта, не вигадуйте.",
        "parameters": _object(
            {
                "order_id": {"type": "string", "pattern": "^[0-9]{5}$"},
                "sku": {"type": "string", "pattern": "^[A-Z0-9-]{2,20}$"},
                "reason": {
                    "type": "string",
                    "enum": ["not_suitable", "defect", "wrong_item", "damaged"],
                },
                "quantity": {"type": "integer", "minimum": 1, "maximum": 50},
                "comment": {"type": "string", "maxLength": 300},
            },
            ["order_id", "sku", "reason", "quantity"],
        ),
    },
]

_TOOL_MAP = {tool["name"]: tool for tool in _TOOLS}
_WORKFLOW_TOOLS = {
    "check_return": {
        "parameters": _object(
            {
                "order_id": {"type": "string", "pattern": "^[0-9]{5}$"},
                "sku": {"type": "string", "pattern": "^[A-Z0-9-]{2,20}$"},
                "reason": {
                    "type": "string",
                    "enum": ["not_suitable", "defect", "wrong_item", "damaged"],
                },
                "quantity": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            ["order_id", "sku", "reason", "quantity"],
        ),
    },
    "check_cancel": {
        "parameters": _object(
            {"order_id": {"type": "string", "pattern": "^[0-9]{5}$"}},
            ["order_id"],
        ),
    },
    "list_returns": {
        "parameters": _object(
            {"order_id": {"type": "string", "pattern": "^[0-9]{5}$"}},
            ["order_id"],
        ),
    },
    "cancel_order": {
        "parameters": _object(
            {"order_id": {"type": "string", "pattern": "^[0-9]{5}$"}},
            ["order_id"],
        ),
    },
}
_TOOL_MAP.update(_WORKFLOW_TOOLS)
_mutation_lock = threading.RLock()


def specs() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": copy.deepcopy(tool["parameters"]),
            },
        }
        for tool in _TOOLS
    ]


def _failure(status: str, code: str, reason: str, arguments: dict | None = None) -> ToolResult:
    return ToolResult(
        status=status,
        content={"error": reason, "code": code},
        reason=reason,
        arguments=arguments,
    )


def _owned_order(order_id: str, ctx: Context) -> dict:
    order = service.get_order(order_id)
    if order["customer_id"] != ctx.customer_id:
        raise PermissionError("Не вдалося підтвердити доступ до цього замовлення.")
    return order


def _public_order(order: dict) -> dict:
    delivery = order.get("delivery", {})
    return {
        "order_id": order["order_id"],
        "created_at": order["created_at"],
        "status": order["status_label"],
        "total": order["total"],
        "items": [
            {
                "sku": item["sku"],
                "name": item["name"],
                "quantity": item["quantity"],
                "price": item["price"],
            }
            for item in order["items"]
        ],
        "delivery": {
            key: delivery.get(key)
            for key in ("method", "city", "point", "tracking", "delivered_at", "cost")
        },
    }


def _execute(name: str, arguments: dict, ctx: Context) -> dict | list:
    if name == "list_orders":
        return service.list_orders(ctx.customer_id)
    if name == "get_order":
        return _public_order(_owned_order(arguments["order_id"], ctx))
    if name == "search_products":
        return service.search_products(**arguments)
    if name == "get_product":
        product = service.get_product(arguments["sku"])
        return {
            key: product[key]
            for key in (
                "sku",
                "name",
                "category",
                "price",
                "weight_kg",
                "warranty_months",
                "returnable",
                "bulky",
                "description",
            )
        }
    if name == "get_stock":
        return service.get_stock(arguments["sku"])
    if name == "delivery_quote":
        return service.delivery_quote(**arguments)
    if name == "create_return":
        _owned_order(arguments["order_id"], ctx)
        existing = service.list_returns(arguments["order_id"])
        duplicate = next(
            (item for item in existing if item["sku"] == arguments["sku"]),
            None,
        )
        if duplicate:
            raise service.PolicyViolation(
                f"Для цього товару вже створено заявку {duplicate['return_id']}."
            )
        record = service.create_return(**arguments)
        return {
            key: record[key]
            for key in (
                "return_id",
                "order_id",
                "sku",
                "quantity",
                "reason",
                "created_at",
                "status",
                "next_steps",
            )
        }
    if name == "check_return":
        order = _owned_order(arguments["order_id"], ctx)
        line = next(
            (item for item in order["items"] if item["sku"] == arguments["sku"]),
            None,
        )
        if line is None:
            raise service.InvalidRequest("Цього товару немає у вказаному замовленні.")
        if arguments["quantity"] > line["quantity"]:
            raise service.InvalidRequest(
                f"У замовленні є {line['quantity']} шт. цього товару."
            )
        product = service.get_product(arguments["sku"])
        issue = None
        if order["status"] != "delivered":
            issue = (
                "Повернення можливе лише після отримання замовлення. "
                f"Поточний статус: {order['status_label']}."
            )
        else:
            delivered = date.fromisoformat(order["delivery"]["delivered_at"])
            elapsed_days = (service.today() - delivered).days
            if arguments["reason"] == "not_suitable":
                if not product["returnable"]:
                    issue = (
                        f"{product['name']} не повертається як товар належної якості."
                    )
                elif elapsed_days > service.RETURN_WINDOW_DAYS:
                    issue = (
                        f"Минуло {elapsed_days} днів; повернення товару належної "
                        f"якості можливе протягом {service.RETURN_WINDOW_DAYS} днів."
                    )
            elif arguments["reason"] == "defect":
                deadline = delivered + timedelta(
                    days=30 * product["warranty_months"]
                )
                if service.today() > deadline:
                    issue = "Гарантійний строк минув."
        existing = service.list_returns(arguments["order_id"])
        duplicate = next(
            (
                item
                for item in existing
                if item["sku"] == arguments["sku"]
                and item["status"] != "rejected"
            ),
            None,
        )
        return {
            "eligible": issue is None and duplicate is None,
            "policy_issue": issue,
            "duplicate": (
                {
                    key: duplicate[key]
                    for key in ("return_id", "order_id", "sku", "status")
                }
                if duplicate
                else None
            ),
            "order_id": order["order_id"],
            "sku": arguments["sku"],
            "quantity": arguments["quantity"],
            "reason": arguments["reason"],
            "order_status": order["status_label"],
            "delivered_at": order["delivery"].get("delivered_at"),
            "warranty_months": product["warranty_months"],
        }
    if name == "check_cancel":
        order = _owned_order(arguments["order_id"], ctx)
        return {
            "eligible": order["status"] == "preparing",
            "order_id": order["order_id"],
            "status": order["status_label"],
            "policy_issue": (
                None
                if order["status"] == "preparing"
                else "Скасувати можна лише замовлення зі статусом «Готується»."
            ),
        }
    if name == "list_returns":
        _owned_order(arguments["order_id"], ctx)
        records = service.list_returns(arguments["order_id"])
        return [
            {
                key: item[key]
                for key in ("return_id", "order_id", "sku", "quantity", "reason", "status")
            }
            for item in records
        ]
    if name == "cancel_order":
        _owned_order(arguments["order_id"], ctx)
        return _public_order(service.cancel_order(arguments["order_id"]))
    raise KeyError(name)


def call(name: str, raw_arguments: str, ctx: Context) -> ToolResult:
    if name not in _TOOL_MAP:
        return _failure("rejected", "unknown_tool", "Ця операція недоступна.")

    try:
        arguments = json.loads(raw_arguments)
    except (TypeError, ValueError):
        return _failure("rejected", "invalid_json", "Аргументи виклику мають бути коректним JSON.")

    if not isinstance(arguments, dict):
        return _failure("rejected", "invalid_arguments", "Аргументи мають бути JSON-об’єктом.")

    errors = list(Draft202012Validator(_TOOL_MAP[name]["parameters"]).iter_errors(arguments))
    if errors:
        error = errors[0]
        field = ".".join(str(part) for part in error.absolute_path)
        reason = f"Некоректні аргументи{f' ({field})' if field else ''}: {error.message}"
        return _failure("rejected", "invalid_arguments", reason, arguments)

    try:
        if name in {"create_return", "cancel_order"}:
            with _mutation_lock:
                content = _execute(name, arguments, ctx)
        else:
            content = _execute(name, arguments, ctx)
        return ToolResult(status="ok", content=content, arguments=arguments)
    except PermissionError as exc:
        return _failure("rejected", "forbidden", str(exc), arguments)
    except service.ShopError as exc:
        logger.warning("Shop tool %s failed: %s", name, exc.code)
        return _failure("error", exc.code, str(exc), arguments)
    except Exception:
        logger.exception("Unexpected error while executing tool %s", name)
        return _failure(
            "error",
            "internal_error",
            "Не вдалося виконати операцію через внутрішню помилку. Спробуйте пізніше.",
            arguments,
        )
