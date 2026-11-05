"""Перевірка середовища для ПР8.

Запуск:
    python check_env.py

Скрипт перевіряє версію Python, наявність потрібних пакетів, налаштування
в `.env`, те, що сервіс магазину читає свої дані, і те, що модель справді
викликає інструменти: описує їй один пробний інструмент, ставить питання,
на яке без нього не відповісти, і показує, що модель запропонувала
викликати, з якими аргументами, скільки це тривало й скільки токенів
коштував сам опис інструмента.

Запустіть перевірку заздалегідь: якщо ключ не працює або обрана модель
не підтримує виклик інструментів, зʼясувати це краще до заняття.

Це діагностика перед роботою, а не зразок для наслідування: тут немає
ані перевірки аргументів, ані виконання, ані повернення результату
моделі — саме це ви проєктуєте самі.
"""

import importlib
import json
import os
import sys
import time
from pathlib import Path

MIN_PYTHON = (3, 10)
PACKAGES = ("openai", "dotenv", "fastapi", "uvicorn", "pydantic")
OPTIONAL_PACKAGES = ("jsonschema",)
SETTINGS = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")
OPTIONAL_SETTINGS = ("LLM_MAX_TOKENS", "LLM_TIMEOUT", "TOOL_MAX_ROUNDS", "SHOP_TODAY")
HERE = Path(__file__).parent

HINTS = {
    "LLM_BASE_URL": "не задано: скопіюйте .env.example у .env",
    "LLM_API_KEY": "не задано: перенесіть ключ із .env попередніх робіт або візьміть "
                   "у Google AI Studio (aistudio.google.com)",
    "LLM_MODEL": "не задано: назву моделі дивіться в Google AI Studio",
}

# Пробний інструмент. Питання про курс валюти на вигадану дату: відповісти
# з памʼяті модель не може, тож мала б попросити інструмент.
PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_exchange_rate",
        "description": "Офіційний курс гривні до іноземної валюти на задану дату.",
        "parameters": {
            "type": "object",
            "properties": {
                "currency": {"type": "string", "enum": ["USD", "EUR", "PLN"],
                             "description": "Код валюти ISO 4217"},
                "date": {"type": "string", "description": "Дата у форматі РРРР-ММ-ДД"},
            },
            "required": ["currency", "date"],
        },
    },
}
PROBE_QUESTION = "Який був офіційний курс євро 3 жовтня 2026 року?"


def report(ok: bool, what: str, hint: str) -> None:
    mark = "[ OK ]" if ok else "[ !! ]"
    tail = f" — {hint}" if hint else ""
    print(f"{mark} {what}{tail}")


def note(what: str, hint: str) -> None:
    """Зауваження, яке не є помилкою середовища."""
    print(f"[ .. ] {what} — {hint}")


def check_python() -> bool:
    actual = sys.version_info[:2]
    ok = actual >= MIN_PYTHON
    need = ".".join(map(str, MIN_PYTHON))
    have = ".".join(map(str, actual))
    report(ok, f"Python {have}", "" if ok else f"потрібен Python {need} або новіший")
    return ok


def check_packages() -> bool:
    ok = True
    for name in PACKAGES:
        try:
            importlib.import_module(name)
        except ImportError:
            report(False, f"пакет {name}", "не встановлено: pip install -r requirements.txt")
            ok = False
        else:
            report(True, f"пакет {name}", "")
    for name in OPTIONAL_PACKAGES:
        try:
            importlib.import_module(name)
        except ImportError:
            note(f"пакет {name}", "не встановлено; потрібен лише для перевірки за JSON Schema")
        else:
            report(True, f"пакет {name}", "")
    return ok


def check_settings() -> bool:
    """Перевірити, що .env заповнений. Значення ключа не друкуємо."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        report(False, "налаштування .env", "перевірку пропущено: немає пакета python-dotenv")
        return False

    if not Path(".env").exists():
        report(False, "файл .env", "немає: скопіюйте .env.example у .env і перенесіть ключ")
    load_dotenv()
    ok = True
    for name in SETTINGS:
        value = os.getenv(name)
        if not value:
            report(False, f"налаштування {name}", HINTS[name])
            ok = False
        elif name == "LLM_API_KEY":
            report(True, f"налаштування {name}", f"задано, довжина {len(value)}")
        else:
            report(True, f"налаштування {name}", value)
    for name in OPTIONAL_SETTINGS:
        value = os.getenv(name)
        if value:
            report(True, f"налаштування {name}", value)
        else:
            note(f"налаштування {name}", "не задано; діє значення за замовчуванням із коду")
    return ok


def check_shop() -> bool:
    """Сервіс магазину читає дані й відповідає."""
    sys.path.insert(0, str(HERE))
    try:
        from shop import service
        customers = service.list_customers()
        orders = service.list_orders(customers[0]["customer_id"])
        products = service.search_products("", limit=50)
    except Exception as exc:  # noqa: BLE001 — діагностика, показуємо все
        report(False, "сервіс магазину", f"не працює ({type(exc).__name__}): {exc} — "
                                         "перевірте, чи повністю розпаковано архів")
        return False
    report(True, "сервіс магазину",
           f"клієнтів {len(customers)}, товарів {len(products)}, "
           f"замовлень першого клієнта {len(orders)}, сьогодні {service.today()}")
    return True


def check_call() -> bool:
    """Описати моделі пробний інструмент і подивитися, чи вона його викличе."""
    try:
        from openai import OpenAI
    except ImportError:
        report(False, "пробний запит", "перевірку пропущено: немає пакета openai")
        return False

    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")
    if not (base_url and api_key and model):
        report(False, "пробний запит", "перевірку пропущено: налаштування неповні")
        return False

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=30)
    messages = [{"role": "user", "content": PROBE_QUESTION}]
    started = time.perf_counter()
    try:
        with_tool = client.chat.completions.create(
            model=model, messages=messages, tools=[PROBE_TOOL], temperature=0,
            max_tokens=max(800, int(os.getenv("LLM_MAX_TOKENS", "800"))),
        )
        without_tool = client.chat.completions.create(
            model=model, messages=messages, temperature=0, max_tokens=200,
        )
    except Exception as exc:  # noqa: BLE001 — діагностика, показуємо все
        report(False, "пробний запит з інструментом", f"збій ({type(exc).__name__}): {exc}")
        return False
    elapsed = time.perf_counter() - started

    choice = with_tool.choices[0]
    calls = choice.message.tool_calls or []
    if not calls:
        text = (choice.message.content or "").strip()
        report(False, "пробний запит з інструментом",
               f"модель відповіла текстом, а не викликом: {text[:80]!r}; "
               f"finish_reason={choice.finish_reason!r} — перевірте, чи "
               "обрана модель підтримує виклик інструментів")
        return False
    for call in calls:
        report(True, "модель запропонувала виклик",
               f"{call.function.name}({call.function.arguments}), id {call.id!r}")
    try:
        args = json.loads(calls[0].function.arguments)
    except ValueError:
        note("аргументи", "рядок аргументів не є JSON — саме такі випадки ви маєте обробляти")
    else:
        if args.get("currency") != "EUR" or args.get("date") != "2026-10-03":
            note("аргументи", f"очікувалось EUR і 2026-10-03, отримано {args} — "
                              "подивіться, як модель тлумачить опис")
    print(f"       finish_reason: {choice.finish_reason!r}; обидва запити — за {elapsed:.2f} с")
    if with_tool.usage and without_tool.usage:
        extra = with_tool.usage.prompt_tokens - without_tool.usage.prompt_tokens
        print(f"       токенів запиту: з описом інструмента {with_tool.usage.prompt_tokens}, "
              f"без нього {without_tool.usage.prompt_tokens} (опис одного інструмента ≈ {extra})")
    return True


def main() -> int:
    print("Перевірка середовища для ПР8\n")
    results = [check_python(), check_packages(), check_settings(), check_shop(), check_call()]
    print()
    if all(results):
        print("Середовище готове до роботи.")
        return 0
    print("Є проблеми — усуньте позначені [ !! ] і запустіть перевірку ще раз.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
