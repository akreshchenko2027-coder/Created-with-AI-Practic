"""Конвеєр обробки документа: від байтів файлу до рішення, що з ним робити.

Єдине місце, яке знає всі кроки послідовно: підготувати зображення
(`images.prepare`), вилучити поля (`llm.extract`), перевірити їх
правилами (`rules.check`) і вирішити долю документа. Учасники одне про
одного не знають: модуль зображень — про модель, модель — про правила,
правила — про веб-рівень.

Рішення про документ — одне з трьох:

* `auto` — дані можна передати в реєстр платежів без людини;
* `review` — дані є, але їх має переглянути людина: причини перелічено;
* `reject` — це не рахунок на оплату або з документа нічого не взяти.

Рішення ухвалює код за результатами перевірок, а не модель. Тут обрано такі
політики:

* які перевірки мають бути пройдені для `auto` і чи всі проблеми
  однаково важкі: неправильна одиниця виміру і неправильний IBAN
  коштують по-різному;
* що робити з полем, якого модель не прочитала, і з полем, яке вона
  прочитала, але про яке сама сказала, що не впевнена;
* що вимірювати окремо: підготовку, вилучення, перевірки.
"""

from dataclasses import dataclass, field
from time import perf_counter

from . import images, llm, rules
from .rules import Issue


@dataclass
class Result:
    """Результат конвеєра — те, з чим працює веб-рівень.

    `decision` — `auto`, `review` або `reject`; `reasons` — чому саме так,
    словами для людини. `document` — поля, які повернула модель і які
    пройшли схему (або `None`, якщо до цього не дійшло). `issues` —
    проблеми, знайдені правилами. `image` — розміри до й після підготовки.
    `elapsed` — час за етапами, наприклад `{"prepare": 0.05,
    "extraction": 6.8, "checks": 0.001}`. `usage` — токени, якщо модель
    викликалась.
    """

    decision: str
    reasons: list[str] = field(default_factory=list)
    document: dict | None = None
    issues: list[Issue] = field(default_factory=list)
    image: dict = field(default_factory=dict)
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None
    raw_response: str | None = None


def decide(document: dict, issues: list[Issue]) -> tuple[str, list[str]]:
    """Вирішити долю документа за вилученими полями й знайденими
    проблемами. Повертає рішення і його причини."""
    document_type = document.get("document_type")
    if document_type and not str(document_type).strip().casefold().startswith("рахунок"):
        return "reject", ["Документ не є рахунком на оплату."]

    def contains_value(value: object) -> bool:
        if isinstance(value, dict):
            return any(contains_value(child) for child in value.values())
        if isinstance(value, list):
            return any(contains_value(child) for child in value)
        return value is not None and value != ""

    has_data = any(
        contains_value(value) for key, value in document.items() if key != "document_type"
    )
    if not has_data:
        return "reject", ["З документа не вдалося вилучити жодного корисного поля."]
    if not issues:
        return "auto", ["Обовʼязкові поля та всі програмні перевірки пройдено."]
    return "review", [f"{issue.field}: {issue.message}" for issue in issues]


def process(content: bytes) -> Result:
    """Обробити файл: підготувати → вилучити → перевірити → вирішити."""
    started = perf_counter()
    prepare_started = perf_counter()
    prepared = images.prepare(content)
    prepare_elapsed = perf_counter() - prepare_started

    extraction_started = perf_counter()
    extracted = llm.extract(prepared)
    extraction_elapsed = perf_counter() - extraction_started
    document = extracted["document"]

    checks_started = perf_counter()
    issues = rules.check(document)
    checks_elapsed = perf_counter() - checks_started
    decision, reasons = decide(document, issues)

    return Result(
        decision=decision,
        reasons=reasons,
        document=document,
        issues=issues,
        image={"original": prepared.original, "sent": prepared.sent},
        model=extracted["model"],
        elapsed={
            "prepare": prepare_elapsed,
            "extraction": extraction_elapsed,
            "checks": checks_elapsed,
            "total": perf_counter() - started,
        },
        usage=extracted["usage"],
        raw_response=extracted["raw_response"],
    )
