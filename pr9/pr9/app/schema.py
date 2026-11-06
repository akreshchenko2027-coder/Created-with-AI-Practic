import json

from jsonschema import Draft202012Validator


_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["return", "cancel", "status", "other", "multiple"],
        },
        "order_id": {"type": ["string", "null"], "pattern": "^[0-9]{5}$"},
        "item": {"type": ["string", "null"], "maxLength": 120},
        "reason": {
            "type": ["string", "null"],
            "enum": ["not_suitable", "defect", "wrong_item", "damaged", None],
        },
        "quantity": {"type": ["integer", "null"], "minimum": 1, "maximum": 50},
        "multiple_actions": {"type": "boolean"},
    },
    "required": ["intent", "order_id", "item", "reason", "quantity", "multiple_actions"],
    "additionalProperties": False,
}


def request_schema() -> dict:
    return _REQUEST_SCHEMA.copy()


def item_schema(items: list[dict]) -> dict:
    skus = [item["sku"] for item in items]
    return {
        "type": "object",
        "properties": {"sku": {"type": ["string", "null"], "enum": [*skus, None]}},
        "required": ["sku"],
        "additionalProperties": False,
    }


def reply_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "reply": {"type": "string", "minLength": 1, "maxLength": 1000},
        },
        "required": ["reply"],
        "additionalProperties": False,
    }


def validate(raw: str) -> dict:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Модель повернула невалідний JSON.") from exc
    errors = sorted(
        Draft202012Validator(_REQUEST_SCHEMA).iter_errors(value),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        error = errors[0]
        field = ".".join(str(part) for part in error.absolute_path)
        raise ValueError(f"Некоректний розбір звернення{f' ({field})' if field else ''}: {error.message}")
    return value


def validate_reply(raw: str) -> dict:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Модель повернула невалідну відповідь у JSON.") from exc
    errors = list(Draft202012Validator(reply_schema()).iter_errors(value))
    if errors:
        error = errors[0]
        raise ValueError(f"Відповідь не пройшла схему: {error.message}")
    return value
