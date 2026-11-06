"""Перевіряє залежності, налаштування, зразки й streaming-провайдера."""

import importlib
import json
import os
import sys
import time
from pathlib import Path

MIN_PYTHON = (3, 10)
PACKAGES = ("openai", "httpx", "dotenv", "fastapi", "uvicorn", "pydantic")
HERE = Path(__file__).parent
STUBS = ("app/providers/openai_compat.py", "app/providers/mock.py", "app/gateway.py",
         "app/drafting.py", "app/jobs.py", "app/limits.py")

HINTS = {
    "LLM_BASE_URL": "не задано: для Ollama використайте http://localhost:11434/v1",
    "LLM_API_KEY": "не задано: для Ollama використайте будь-який непорожній рядок",
    "LLM_MODEL": "не задано: вкажіть модель, установлену в Ollama",
}

PROBE_MESSAGES = [
    {"role": "system", "content": "Ти — оператор підтримки інтернет-магазину. Відповідай українською, "
                                  "коротко й ввічливо."},
    {"role": "user", "content": "Клієнт пише: «Де моє замовлення 10458?». Відомо: передано Новій пошті "
                                "4 жовтня, накладна 20450012345678, відділення № 12 у Києві. "
                                "Склади відповідь клієнтові з трьох речень."},
]


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
    return ok


def check_settings() -> tuple[bool, str]:
    """Перевірити `.env`. Повертає (ok, провайдер). Значення ключа не
    друкуємо."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        report(False, "налаштування .env", "перевірку пропущено: немає пакета python-dotenv")
        return False, ""

    if not Path(".env").exists():
        report(False, "файл .env", "немає: скопіюйте .env.example у .env і перенесіть ключ")
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "openai")
    fallback = os.getenv("LLM_FALLBACK", "none")
    report(provider in ("openai", "mock"), f"провайдер {provider}",
           "" if provider in ("openai", "mock") else "очікується openai або mock")
    report(True, f"запасний варіант {fallback}", "")

    ok = True
    if provider == "openai" or fallback not in ("none", "mock"):
        for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
            value = os.getenv(name)
            if not value:
                report(False, f"налаштування {name}", HINTS[name])
                ok = False
            elif name == "LLM_API_KEY":
                report(True, f"налаштування {name}", f"задано, довжина {len(value)}")
            else:
                report(True, f"налаштування {name}", value)
    else:
        note("ключ доступу", "не потрібен, поки провайдер — mock; справжнього провайдера "
                             "перевірте, коли переключите LLM_PROVIDER=openai")
    for name in ("LLM_TIMEOUT", "LLM_IDLE_TIMEOUT", "LLM_TOTAL_TIMEOUT", "LLM_MAX_ATTEMPTS",
                 "LLM_MAX_CONCURRENT", "RATE_LIMIT_PER_MINUTE"):
        value = os.getenv(name)
        if value:
            report(True, f"налаштування {name}", value)
        else:
            note(f"налаштування {name}", "не задано; діє значення за замовчуванням із коду")
    return ok, provider


def check_samples() -> bool:
    path = HERE / "samples" / "requests.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        ids = [s["id"] for s in data]
        missing = [s["id"] for s in data if not s.get("facts") or not s.get("reply_example")]
    except (OSError, ValueError, KeyError) as exc:
        report(False, "зразки звернень", f"не читаються ({type(exc).__name__}): {exc} — "
                                          "перевірте, чи повністю розпаковано архів")
        return False
    report(not missing, f"зразки звернень: {len(ids)} ({', '.join(ids)})",
           "" if not missing else f"без фактів або прикладу відповіді: {', '.join(missing)}")
    return not missing


def check_stubs() -> None:
    """Які модулі ще заготовки. Це не помилка середовища — так і має бути
    на початку роботи."""
    pending = []
    for rel in STUBS:
        text = (HERE / rel).read_text(encoding="utf-8", errors="replace")
        if "NotImplementedError" in text:
            pending.append(rel)
    if pending:
        note("заготовки", "ще не реалізовано: " + ", ".join(pending))
    else:
        report(True, "заготовки", "усі модулі мають реалізацію")


def check_call(provider: str) -> bool:
    """Пробний запит у режимі streaming до справжнього провайдера."""
    if provider != "openai":
        note("пробний запит", "пропущено: провайдер — mock")
        return True
    try:
        import openai
    except ImportError:
        report(False, "пробний запит", "перевірку пропущено: немає пакета openai")
        return False

    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")
    if not (base_url and api_key and model):
        report(False, "пробний запит", "перевірку пропущено: налаштування неповні")
        return False

    timeout = float(os.getenv("LLM_TIMEOUT", "20"))
    client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)
    started = time.perf_counter()
    first = None
    chunks = 0
    text = []
    try:
        stream = client.chat.completions.create(
            model=model, messages=PROBE_MESSAGES, temperature=0.3, max_tokens=200, stream=True,
        )
        for chunk in stream:
            if first is None:
                first = time.perf_counter() - started
            chunks += 1
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                text.append(delta)
    except openai.APITimeoutError:
        report(False, "пробний запит", f"таймаут після {timeout:.0f} с — збій виду timeout; "
                                       "збільште LLM_TIMEOUT або перевірте мережу")
        return False
    except openai.RateLimitError as exc:
        retry_after = exc.response.headers.get("retry-after") if exc.response is not None else None
        report(False, "пробний запит", f"429, ліміт провайдера — збій виду rate_limit; "
                                       f"Retry-After: {retry_after or 'не надіслано'}")
        return False
    except openai.AuthenticationError:
        report(False, "пробний запит", "401 — збій виду auth: ключ невірний або протермінований; "
                                       "повторювати такий запит не має сенсу")
        return False
    except openai.APIConnectionError as exc:
        report(False, "пробний запит", f"немає зʼєднання — збій виду unavailable: {exc}")
        return False
    except openai.APIStatusError as exc:
        kind = "unavailable" if exc.status_code >= 500 else "bad_request"
        report(False, "пробний запит", f"статус {exc.status_code} — збій виду {kind}: {exc}")
        return False
    except Exception as exc:
        report(False, "пробний запит", f"збій ({type(exc).__name__}): {exc}")
        return False

    total = time.perf_counter() - started
    answer = "".join(text).strip()
    if not answer:
        report(False, "пробний запит", f"потік завершився без тексту ({chunks} фрагментів)")
        return False
    report(True, "пробний запит у режимі streaming",
           f"перший фрагмент через {first:.2f} с, разом {total:.2f} с, фрагментів {chunks}")
    print(f"       {answer[:160]}{'…' if len(answer) > 160 else ''}")
    note("таймаути", f"LLM_TIMEOUT={timeout:.0f} с на спробу; перший фрагмент прийшов за {first:.1f} с — "
                     "від цього відштовхуйтеся, обираючи LLM_IDLE_TIMEOUT і LLM_TOTAL_TIMEOUT")
    return True


def main() -> int:
    print("Перевірка середовища для ПР10\n")
    results = [check_python(), check_packages()]
    settings_ok, provider = check_settings()
    results += [settings_ok, check_samples()]
    check_stubs()
    results.append(check_call(provider))
    print()
    if all(results):
        print("Середовище готове до роботи.")
        return 0
    print("Є проблеми — усуньте позначені [ !! ] і запустіть перевірку ще раз.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
