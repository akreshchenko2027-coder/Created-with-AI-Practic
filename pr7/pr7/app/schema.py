"""Контракт відповіді моделі: які поля вона має повернути й як це
перевіряється.

Окремий модуль, як у ПР4 і ПР6: схема — домовленість між моделлю та
рештою застосунку, і від провайдера вона не залежить. `app/rules.py` і
`app/extraction.py` працюють лише з тим, що пройшло перевірку тут.

Набір полів задано (розділ 2 практичної роботи):

* тип документа — рахунок на оплату чи інший документ;
* номер рахунку, дата, строк дії (якщо є);
* постачальник: назва, код (ЄДРПОУ або РНОКПП), IBAN;
* покупець: назва, код;
* позиції: найменування, одиниця, кількість, ціна, сума;
* разом без ПДВ, ПДВ, усього до сплати.

Рішення — ваші:

* як назвати поля й що з них обовʼязкове в схемі;
* у якому вигляді повертати дату, суму, IBAN — як надруковано чи
  нормалізовано, і хто нормалізує: модель чи ваш код;
* як модель позначає значення, якого в документі немає, і значення,
  яке є, але не читається, — і чи це одне й те саме;
* чи просити від моделі щось понад задані поля (упевненість, фрагмент
  тексту, з якого взято значення) — і чи буде кому цим скористатися.

Описати схему можна JSON Schema вручну або pydantic-моделлю.
"""

import json
import logging
import re
from datetime import date
from decimal import Decimal, InvalidOperation

from jsonschema import Draft202012Validator, FormatChecker

logger = logging.getLogger(__name__)


def _normalize_dates(
    value: object,
    key: str | None = None,
    path: tuple[str, ...] = (),
    warnings: list[str] | None = None,
) -> object:
    if isinstance(value, dict):
        return {
            name: _normalize_dates(child, name, (*path, name), warnings)
            for name, child in value.items()
        }
    if isinstance(value, list):
        return [
            _normalize_dates(child, path=(*path, str(index)), warnings=warnings)
            for index, child in enumerate(value)
        ]
    if key not in {"date", "valid_until"} or not isinstance(value, str):
        return value

    text = value.strip()
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        pass

    match = re.fullmatch(r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})", text)
    if match:
        day, month, year = map(int, match.groups())
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            pass

    match = re.fullmatch(r"(\d{4})[./](\d{1,2})[./](\d{1,2})", text)
    if match:
        year, month, day = map(int, match.groups())
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            pass

    if warnings is not None:
        warnings.append(".".join(path))
    return None


def _normalize_amounts(value: object, key: str | None = None) -> object:
    if isinstance(value, str) and value.strip().casefold() == "null":
        return None
    if isinstance(value, dict):
        return {name: _normalize_amounts(child, name) for name, child in value.items()}
    if isinstance(value, list):
        return [_normalize_amounts(child) for child in value]
    if key not in {"price", "amount", "total_without_vat", "vat", "total"}:
        return value
    if value is None or isinstance(value, bool):
        return value
    if not isinstance(value, (int, float, str)):
        return value

    text = re.sub(r"\s+", "", str(value))
    if not re.fullmatch(r"[+-]?\d+(?:[.,]\d+)*", text):
        return value

    if "," in text and "." in text:
        decimal_separator = "," if text.rfind(",") > text.rfind(".") else "."
        grouping_separator = "." if decimal_separator == "," else ","
        text = text.replace(grouping_separator, "")
        if decimal_separator == ",":
            text = text.replace(",", ".")
    elif "," in text:
        text = text.replace(",", ".")
    elif text.count(".") > 1:
        text = text.replace(".", "")
    elif "." in text and len(text.rsplit(".", 1)[1]) == 3:
        text = text.replace(".", "")

    try:
        amount = Decimal(text)
    except InvalidOperation:
        return value
    normalized = format(amount, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def _allows_null(node: dict) -> bool:
    value_type = node.get("type")
    if value_type == "null":
        return True
    if isinstance(value_type, list) and "null" in value_type:
        return True
    return any(
        isinstance(option, dict) and _allows_null(option)
        for option in node.get("oneOf", [])
    )


def _fill_missing_fields(
    value: object,
    node: dict,
    path: tuple[str, ...] = (),
    warnings: list[str] | None = None,
) -> object:
    if isinstance(value, dict) and node.get("type") == "object":
        properties = node.get("properties", {})
        for name in node.get("required", []):
            child_path = (*path, name)
            if name not in value:
                child_schema = properties[name]
                if _allows_null(child_schema):
                    value[name] = None
                    if warnings is not None:
                        warnings.append(".".join(child_path))
                elif child_schema.get("type") == "object":
                    value[name] = {}
                else:
                    continue
            value[name] = _fill_missing_fields(
                value[name], properties[name], child_path, warnings
            )
    elif isinstance(value, list) and isinstance(node.get("items"), dict):
        return [
            _fill_missing_fields(child, node["items"], (*path, str(index)), warnings)
            for index, child in enumerate(value)
        ]
    return value


def output_schema() -> dict:
    """Повернути JSON Schema відповіді моделі.

    Та сама схема передається моделі як опис очікуваного результату
    (через `response_format` або текстом в інструкції) і використовується
    для перевірки того, що повернулося.
    """
    nullable_text = {"type": ["string", "null"]}
    nullable_amount = {
        "oneOf": [
            {"type": "string", "pattern": r"^-?\d+(?:[.,]\d{1,2})?$"},
            {"type": "null"},
        ]
    }

    def object_schema(properties: dict) -> dict:
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }

    item = object_schema({
        "name": nullable_text,
        "unit": nullable_text,
        "quantity": {"type": ["number", "null"]},
        "price": nullable_amount,
        "amount": nullable_amount,
    })
    return object_schema({
        "document_type": nullable_text,
        "number": nullable_text,
        "date": {
            "oneOf": [
                {"type": "string", "format": "date"},
                {"type": "null"},
            ]
        },
        "valid_until": {
            "oneOf": [
                {"type": "string", "format": "date"},
                {"type": "null"},
            ]
        },
        "supplier": object_schema({
            "name": nullable_text,
            "code": nullable_text,
            "iban": nullable_text,
        }),
        "buyer": object_schema({
            "name": nullable_text,
            "code": nullable_text,
        }),
        "items": {"type": ["array", "null"], "items": item},
        "total_without_vat": nullable_amount,
        "vat": nullable_amount,
        "total": nullable_amount,
    })


def validate(raw: str) -> dict:
    """Перевірити сиру відповідь моделі й повернути дані, яким можна
    довіряти структурно.

    Відсутні обовʼязкові значення й некоректні дати стають null з
    попередженням у журналі; це дає правилам позначити документ для перевірки.
    Якщо текст не є JSON або інша частина відповіді не проходить схему —
    підняти помилку з поясненням, що саме не так.

    Схема перевіряє форму: суми — десяткові рядки, дата — дата, позиції — список.
    Чи сходиться сума з позиціями і чи існує такий IBAN — перевірка
    змісту, і її місце в `app/rules.py`.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Модель повернула невалідний JSON: {exc}") from exc

    warnings: list[str] = []
    data = _normalize_dates(data, warnings=warnings)
    data = _normalize_amounts(data)
    data = _fill_missing_fields(data, output_schema(), warnings=warnings)
    if warnings:
        logger.warning(
            "Model response contained invalid or missing values; replaced with null: %s",
            ", ".join(sorted(warnings)),
        )
    validator = Draft202012Validator(output_schema(), format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(data), key=lambda error: list(map(str, error.path)))
    if errors:
        error = errors[0]
        path = ".".join(str(part) for part in error.absolute_path) or "відповідь"
        raise ValueError(f"Відповідь не відповідає схемі ({path}): {error.message}")
    return data
