"""Вартість виклику: токени → гроші.

Провайдер тарифікує токени окремо на вході й на виході; кешовані ним
вхідні токени — дешевше. Ціни — з `.env`, за мільйон токенів; значення
зі зразка орієнтовні, чинні — на сторінці цін провайдера. Безкоштовний
рівень не скасовує розрахунку: рахуємо так, ніби платимо, інакше нема
чого обмежувати й порівнювати.

Модуль знає лише ціни та `usage`; про провайдерів, чернетки й журнал
він не знає. Шлюз викликає `estimate` після кожного звертання і кладе
результат у `Result.cost` та в журнал; ліміт (`budget.py`) додає його
до витрат дня.

Функція нижче — заготовка. Рішення з розділу 2 практичної роботи:

* назви полів `usage`: OpenAI-сумісний API віддає `prompt_tokens`,
  `completion_tokens`, іноді `prompt_tokens_details.cached_tokens`;
  підставний провайдер — те, що ви в ньому напишете. Що робити з
  полями, яких немає;
* `usage` не прийшов (потік обірвано, провайдер не підтримує
  `stream_options`): повернути `None`, нуль чи оцінку за довжиною
  тексту (приблизно 4 символи на токен для англійської, для
  української помітно менше)? Якщо оцінку — вона має бути позначена
  як оцінка, а не факт, у `Result` і в журналі;
* чия ціна: модель запасного варіанту може коштувати інакше, ніж
  основна; підставний провайдер за замовчуванням рахується за цінами
  справжнього (`MOCK_BILLABLE=1`), щоб бюджет можна було перевірити
  без ключа — чи лишати так;
* одиниці й округлення: долари з шістьма знаками, центи, мікродолари?
  Суми за день складаються з тисяч дрібних чисел.
"""

import math
import os

from dotenv import load_dotenv

load_dotenv()

# Ціни за мільйон токенів, у доларах.
PRICE_INPUT = float(os.getenv("LLM_PRICE_INPUT_PER_1M", "0.30"))
PRICE_OUTPUT = float(os.getenv("LLM_PRICE_OUTPUT_PER_1M", "2.50"))
_cached = os.getenv("LLM_PRICE_CACHED_INPUT_PER_1M", "")
PRICE_CACHED_INPUT = float(_cached) if _cached else PRICE_INPUT * 0.25
MOCK_BILLABLE = os.getenv("MOCK_BILLABLE", "1") == "1"
if any(not math.isfinite(price) or price < 0 for price in
       (PRICE_INPUT, PRICE_OUTPUT, PRICE_CACHED_INPUT)):
    raise ValueError("Ціни мають бути скінченними невідʼємними числами.")


def _tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 2.5)) if text else 0


def _is_billable(provider: str | None) -> bool:
    return provider != "mock" or MOCK_BILLABLE


def settings() -> dict:
    """Чинні ціни — для сторінки й журналу."""
    return {
        "input_per_1m": PRICE_INPUT,
        "output_per_1m": PRICE_OUTPUT,
        "cached_input_per_1m": PRICE_CACHED_INPUT,
        "mock_billable": MOCK_BILLABLE,
        "currency": "USD",
    }


def estimate(usage: dict | None, *, provider: str | None = None, model: str | None = None,
             text: str | None = None, prompt: str | None = None) -> tuple[float | None, bool]:
    """Оцінити вартість одного звертання до провайдера.

    `usage` — те, що повернув провайдер (може бути `None`); `provider` і
    `model` — хто відповідав; `text` — текст відповіді, якщо доведеться
    оцінювати без `usage`.

    Повертає пару: вартість у доларах (або `None`, якщо її не визначити)
    і ознаку «це оцінка, а не розрахунок за фактичним `usage`».
    """
    if not _is_billable(provider):
        return 0.0, False

    estimated = usage is None
    usage = usage if isinstance(usage, dict) else {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int) or prompt_tokens < 0:
        prompt_tokens = _tokens(prompt or "")
        estimated = True
    if isinstance(completion_tokens, bool) or not isinstance(completion_tokens, int) or completion_tokens < 0:
        completion_tokens = _tokens(text or "")
        estimated = True
    details = usage.get("prompt_tokens_details")
    cached_tokens = details.get("cached_tokens", 0) if isinstance(details, dict) else 0
    if isinstance(cached_tokens, bool) or not isinstance(cached_tokens, int) or cached_tokens < 0:
        cached_tokens = 0
        estimated = True
    cached_tokens = min(cached_tokens, prompt_tokens)
    amount = (
        (prompt_tokens - cached_tokens) * PRICE_INPUT
        + cached_tokens * PRICE_CACHED_INPUT
        + completion_tokens * PRICE_OUTPUT
    ) / 1_000_000
    return round(amount, 12), estimated


def maximum_cost(prompt: str, max_output_tokens: int, *,
                 provider: str | None = None) -> float:
    if not _is_billable(provider):
        return 0.0
    return round((_tokens(prompt) * PRICE_INPUT + max_output_tokens * PRICE_OUTPUT) / 1_000_000, 12)
