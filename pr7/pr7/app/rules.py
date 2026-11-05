"""Програмні правила: перевірка змісту вилучених даних.

Схема (`app/schema.py`) каже, що IBAN — рядок, а сума — число. Чи це
справжній IBAN і чи сходиться сума з позиціями, схема не знає. Це знає
документ: у рахунку багато надлишковості — контрольні розряди в кодах,
підсумки, що мають дорівнювати сумі рядків, ПДВ, що має дорівнювати
п'ятій частині. Модель, яка прочитала 8 замість 3, цю надлишковість
порушує, і код це бачить. Правила тут не знають ні про модель, ні про
веб-рівень: на вході — перевірений за схемою словник, на виході — перелік
знайдених проблем.

Набір правил задано (розділ 2 практичної роботи):

* IBAN: формат і контрольне число за ISO 13616 (залишок mod 97);
* код постачальника й покупця: 8 цифр ЄДРПОУ з контрольним розрядом
  або 10 цифр РНОКПП;
* арифметика: кількість × ціна = сума в кожному рядку; сума рядків =
  разом без ПДВ; ПДВ = 20 % (або нуль для неплатника); разом + ПДВ =
  до сплати — з допуском на округлення, який ви оберете й обґрунтуєте;
* дати: існують, не з майбутнього, строк дії не раніше дати рахунку;
* покупець — ваша компанія (`reference/company.json`);
* постачальник відомий, і IBAN збігається з довідником
  (`reference/suppliers.json`);
* обовʼязкові поля присутні.

Рішення — ваші: які правила дають помилку, а які — попередження; чи
перевіряти правило, якщо потрібного поля немає; як рахувати гроші, щоб
0,1 + 0,2 не дало 0,30000000000000004.
"""

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"


@dataclass
class Issue:
    """Проблема, знайдена правилом.

    `field` — шлях до поля, наприклад `supplier.iban` чи `items.1.amount`
    (другий рядок: індекс — з нуля); сторінка підсвічує поле з таким
    шляхом. `rule` — коротка назва
    правила. `message` — пояснення для людини, яка перевірятиме документ.
    `severity` — `error` або `warning`; що з цього випливає для рішення,
    вирішує `app/extraction.py`.
    """

    field: str
    rule: str
    message: str
    severity: str = "error"


def load_reference() -> tuple[dict, list[dict]]:
    """Прочитати реквізити своєї компанії й довідник постачальників."""
    company = json.loads((REFERENCE_DIR / "company.json").read_text(encoding="utf-8"))
    suppliers = json.loads((REFERENCE_DIR / "suppliers.json").read_text(encoding="utf-8"))
    return company, suppliers


def check(document: dict) -> list[Issue]:
    """Застосувати правила до вилученого документа.

    Порожній перелік означає, що правила нічого не знайшли, — а не що
    документ прочитано правильно. Різниця між цими двома речами і є
    предметом перевірки на наборі зразків.
    """
    issues: list[Issue] = []

    def add(field: str, rule: str, message: str) -> None:
        issues.append(Issue(field, rule, message))

    def clean(value: object) -> str:
        text = unicodedata.normalize("NFKC", str(value or "")).casefold()
        return " ".join(
            text.replace("’", "'").replace("ʼ", "'").replace("`", "'").split()
        )

    def money(value: object) -> Decimal | None:
        if value is None or value == "":
            return None
        text = str(value).replace("\u00a0", "").replace(" ", "").replace(",", ".")
        try:
            return Decimal(text)
        except InvalidOperation:
            return None

    def valid_iban(value: object) -> bool:
        iban = re.sub(r"\s+", "", str(value or "")).upper()
        if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", iban):
            return False
        if iban.startswith("UA") and len(iban) != 29:
            return False
        remainder = 0
        for char in iban[4:] + iban[:4]:
            digits = str(ord(char) - 55) if char.isalpha() else char
            for digit in digits:
                remainder = (remainder * 10 + int(digit)) % 97
        return remainder == 1

    def valid_code(value: object) -> bool:
        code = str(value or "")
        if re.fullmatch(r"\d{10}", code):
            weights = (-1, 5, 7, 9, 4, 6, 10, 5, 7)
            check_digit = sum(int(char) * weight for char, weight in
                              zip(code[:9], weights)) % 11
            return check_digit % 10 == int(code[9])
        if not re.fullmatch(r"\d{8}", code):
            return False
        weights = ([1, 2, 3, 4, 5, 6, 7] if int(code[0]) <= 2
                   else [7, 1, 2, 3, 4, 5, 6])
        digit = sum(int(char) * weight for char, weight in zip(code[:7], weights)) % 11
        if digit == 10:
            digit = sum(int(char) * weight for char, weight in
                        zip(code[:7], range(3, 10))) % 11
        return digit % 10 == int(code[7])

    doc_type = clean(document.get("document_type"))
    if not doc_type.startswith("рахунок"):
        add("document_type", "document_type", "Документ не визначено як рахунок на оплату.")

    required = (
        "number", "date", "supplier.name", "supplier.code", "supplier.iban",
        "buyer.name", "buyer.code", "items", "total_without_vat", "vat", "total",
    )
    for path in required:
        value: object = document
        for part in path.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if value is None or value == "" or value == []:
            add(path, "missing_required", f"Обовʼязкове поле «{path}» відсутнє або не читається.")

    for index, item in enumerate(document.get("items") or []):
        for field_name in ("name", "unit", "quantity", "price", "amount"):
            if item.get(field_name) is None or item.get(field_name) == "":
                add(f"items.{index}.{field_name}", "missing_required",
                    f"У позиції {index + 1} відсутнє поле «{field_name}».")

    supplier = document.get("supplier") or {}
    buyer = document.get("buyer") or {}
    company, suppliers = load_reference()

    iban = supplier.get("iban")
    if iban and not valid_iban(iban):
        add("supplier.iban", "iban_checksum", "IBAN має неправильний формат або контрольне число.")

    code = supplier.get("code")
    if code and not valid_code(code):
        add("supplier.code", "code_checksum", "Код постачальника має некоректну контрольну цифру.")

    if buyer.get("code") and not valid_code(buyer["code"]):
        add("buyer.code", "code_checksum", "Код покупця має некоректну контрольну цифру.")
    if buyer.get("code") and str(buyer["code"]) != str(company["code"]):
        add("buyer.code", "buyer", "Рахунок виставлено не компанії «Сузірʼя Рітейл».")
    elif buyer.get("name") and clean(buyer["name"]) != clean(company["name"]):
        add("buyer.name", "buyer", "Назва покупця не збігається з нашою компанією.")

    supplier_by_code = next(
        (entry for entry in suppliers if code and str(entry["code"]) == str(code)),
        None,
    )
    supplier_by_name = next(
        (entry for entry in suppliers if supplier.get("name")
         and clean(entry["name"]) == clean(supplier["name"])),
        None,
    )
    supplier_match = supplier_by_code or supplier_by_name
    if supplier.get("name") or code:
        if supplier_match is None:
            add("supplier.code", "supplier_registry", "Постачальника немає у довіднику.")
        else:
            if code and str(code) != str(supplier_match["code"]):
                add("supplier.code", "supplier_registry", "Код не збігається з довідником постачальників.")
            if (supplier.get("name")
                    and clean(supplier["name"]) != clean(supplier_match["name"])):
                add("supplier.name", "supplier_registry", "Назва не збігається з довідником постачальників.")
            if iban and re.sub(r"\s+", "", str(iban)).upper() != supplier_match["iban"]:
                add("supplier.iban", "iban_registry", "IBAN не збігається з довідником постачальників.")

    parsed_dates: dict[str, date] = {}
    for field_name in ("date", "valid_until"):
        value = document.get(field_name)
        if value is None:
            continue
        try:
            parsed_dates[field_name] = date.fromisoformat(value)
        except (TypeError, ValueError):
            add(field_name, "date_format", "Дата має бути коректною датою у форматі YYYY-MM-DD.")
    if parsed_dates.get("date") and parsed_dates["date"] > date.today():
        add("date", "date_future", "Дата рахунку не може бути в майбутньому.")
    if (parsed_dates.get("date") and parsed_dates.get("valid_until")
            and parsed_dates["valid_until"] < parsed_dates["date"]):
        add("valid_until", "date_order", "Строк дії не може передувати даті рахунку.")

    item_amounts: list[Decimal] = []
    for index, item in enumerate(document.get("items") or []):
        quantity = money(item.get("quantity"))
        price = money(item.get("price"))
        amount = money(item.get("amount"))
        if quantity is None or price is None or amount is None:
            continue
        item_amounts.append(amount)
        calculated = (quantity * price).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if calculated != amount:
            add(f"items.{index}.amount", "arithmetic_line",
                f"Сума позиції {amount} не дорівнює кількості × ціні ({calculated}).")

    subtotal = money(document.get("total_without_vat"))
    vat = money(document.get("vat"))
    total = money(document.get("total"))
    if subtotal is not None and item_amounts:
        item_total = sum(item_amounts, Decimal("0.00")).quantize(Decimal("0.01"))
        if item_total != subtotal:
            add("total_without_vat", "arithmetic_subtotal",
                f"Сума позицій {item_total} не дорівнює підсумку без ПДВ {subtotal}.")
    if subtotal is not None and vat is not None:
        expected_total = (subtotal + vat).quantize(Decimal("0.01"))
        if total is not None and expected_total != total:
            add("total", "arithmetic_total",
                f"Підсумок з ПДВ має дорівнювати {expected_total}.")
        if supplier_match and supplier_match.get("vat_payer"):
            expected_vat = (subtotal * Decimal("0.20")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            if abs(vat - expected_vat) > Decimal("0.01"):
                add("vat", "vat_rate", f"ПДВ не дорівнює 20% підсумку без ПДВ ({expected_vat}).")
        elif supplier_match and not supplier_match.get("vat_payer") and vat != 0:
            add("vat", "vat_rate", "У довіднику постачальник зазначений як неплатник ПДВ.")

    return issues
