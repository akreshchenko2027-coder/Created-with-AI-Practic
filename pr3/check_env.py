"""Перевірка середовища для ПР3.

Запуск:
    python check_env.py

Скрипт перевіряє версію Python, наявність потрібних пакетів, заповненість
налаштувань у файлі .env і те, що модель справді відповідає — надсилає
короткий пробний запит і показує час відповіді.

Запустіть перевірку заздалегідь: якщо ключ не працює або сервіс недоступний,
зʼясувати це краще до заняття.

Це діагностика перед роботою, а не зразок для наслідування: тут немає ані
системної інструкції, ані контексту, ані обробки збоїв — саме це ви
проєктуєте самі.
"""

import importlib
import os
import sys
import time

MIN_PYTHON = (3, 10)
PACKAGES = ("openai", "dotenv", "fastapi", "uvicorn")
SETTINGS = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")

HINTS = {
    "LLM_BASE_URL": "не задано: скопіюйте .env.example у .env",
    "LLM_API_KEY": "не задано: візьміть ключ у Google AI Studio (aistudio.google.com) "
                   "і впишіть його в .env",
    "LLM_MODEL": "не задано: назву моделі дивіться в Google AI Studio",
}


def report(ok: bool, what: str, hint: str) -> None:
    mark = "[ OK ]" if ok else "[ !! ]"
    tail = f" — {hint}" if hint else ""
    print(f"{mark} {what}{tail}")


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
    return ok


def check_settings() -> bool:
    """Перевірити, що .env заповнений. Значення ключа не друкуємо."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        report(False, "налаштування .env", "перевірку пропущено: немає пакета python-dotenv")
        return False

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
    return ok


def check_call() -> bool:
    """Надіслати короткий пробний запит і зміряти час відповіді."""
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
    started = time.perf_counter()
    try:
        answer = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Відповідай одним словом: працюєш?"}],
            max_tokens=20,
        )
    except Exception as exc:
        report(False, "пробний запит", f"збій ({type(exc).__name__}): {exc}")
        return False
    elapsed = time.perf_counter() - started

    text = (answer.choices[0].message.content or "").strip()
    usage = getattr(answer, "usage", None)
    report(True, "пробний запит", f"відповідь за {elapsed:.2f} с: {text[:60]!r}")
    if usage:
        print(f"       токенів: запит {usage.prompt_tokens}, "
              f"відповідь {usage.completion_tokens}, разом {usage.total_tokens}")
    return True


def main() -> int:
    print("Перевірка середовища для ПР3\n")
    results = [check_python(), check_packages(), check_settings(), check_call()]
    print()
    if all(results):
        print("Середовище готове до роботи.")
        return 0
    print("Є проблеми — усуньте позначені [ !! ] і запустіть перевірку ще раз.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
