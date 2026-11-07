"""Перевірка середовища для ПР11.

Запуск:
    python check_env.py

Скрипт перевіряє версію Python, наявність потрібних пакетів,
налаштування в `.env`, зразки звернень, чи можна писати в папки журналу
й стану, чи перенесено модулі з ПР10 і які модулі каркаса ще заготовки,
і — якщо обрано справжнього провайдера — робить пробний запит у режимі
streaming: дивиться, чи віддає провайдер `usage` в потоці (від цього
залежить, чи буде вартість розрахунком або оцінкою), і рахує вартість
пробного запиту за цінами з `.env`.

Це діагностика перед роботою, а не зразок для наслідування.
"""

import importlib
import json
import os
import sys
import time
from pathlib import Path

MIN_PYTHON = (3, 10)
PACKAGES = ("openai", "dotenv", "fastapi", "uvicorn", "pydantic")
HERE = Path(__file__).parent
FROM_PR10 = ("app/providers/openai_compat.py", "app/providers/mock.py", "app/gateway.py",
             "app/jobs.py", "app/limits.py", "app/drafting.py")
STUB_MARK = "Заготовка з ПР10"
NEW_STUBS = ("app/costs.py", "app/cache.py", "app/budget.py", "eval/analyze.py")

HINTS = {
    "LLM_BASE_URL": "не задано: скопіюйте .env.example у .env",
    "LLM_API_KEY": "не задано: перенесіть ключ із .env попередніх робіт або візьміть "
                   "у Google AI Studio (aistudio.google.com)",
    "LLM_MODEL": "не задано: назву моделі дивіться в Google AI Studio",
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

    ok = True
    if not Path(".env").exists():
        report(False, "файл .env", "немає: скопіюйте .env.example у .env і перенесіть ключ")
        ok = False
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "openai")
    fallback = os.getenv("LLM_FALLBACK", "none")
    report(provider in ("openai", "ollama", "mock"), f"провайдер {provider}",
           "" if provider in ("openai", "ollama", "mock") else "очікується openai, ollama або mock")
    report(True, f"запасний варіант {fallback}", "")

    if provider in ("openai", "ollama") or fallback not in ("none", "mock"):
        required_settings = ("LLM_BASE_URL", "LLM_MODEL")
        if provider != "ollama":
            required_settings += ("LLM_API_KEY",)
        for name in required_settings:
            value = os.getenv(name)
            if not value:
                report(False, f"налаштування {name}", HINTS[name])
                ok = False
            elif name == "LLM_API_KEY":
                report(True, f"налаштування {name}", f"задано, довжина {len(value)}")
            else:
                report(True, f"налаштування {name}", value)
    else:
        note("ключ доступу", "не потрібен, поки провайдер — mock")

    for name in ("LLM_PRICE_INPUT_PER_1M", "LLM_PRICE_OUTPUT_PER_1M"):
        value = os.getenv(name)
        if value:
            try:
                float(value)
                report(True, f"налаштування {name}", f"{value} $ за мільйон токенів")
            except ValueError:
                report(False, f"налаштування {name}", f"не число: {value}")
                ok = False
        else:
            note(f"налаштування {name}", "не задано; діє значення за замовчуванням із app/costs.py")
    for name in ("CACHE_ENABLED", "CACHE_TTL_SECONDS", "BUDGET_DAILY_USD", "BUDGET_WARN_AT",
                 "RATE_LIMIT_PER_MINUTE", "LOG_LEVEL"):
        value = os.getenv(name)
        if value:
            report(True, f"налаштування {name}", value)
        else:
            note(f"налаштування {name}", "не задано; діє значення за замовчуванням із коду")
    return ok, provider


def check_dirs() -> bool:
    """Папки журналу й стану мають створюватися й бути доступними для
    запису."""
    ok = True
    log_file = Path(os.getenv("LOG_FILE", "logs/app.jsonl"))
    targets = {"журнал": log_file.parent}
    for name in ("CACHE_FILE", "BUDGET_FILE"):
        value = os.getenv(name, "state/x.json")
        if value:
            targets[f"стан ({name})"] = Path(value).parent
    for label, directory in targets.items():
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / ".write-test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            report(True, f"{label}: папка {directory}", "доступна для запису")
        except OSError as exc:
            report(False, f"{label}: папка {directory}", f"не вдалося записати: {exc}")
            ok = False
    return ok


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


def check_transfer() -> None:
    """Чи перенесено модулі з ПР10. Заготовка каркаса містить позначку й
    `NotImplementedError`; свій файл з ПР10 — ні."""
    pending = []
    for rel in FROM_PR10:
        text = (HERE / rel).read_text(encoding="utf-8", errors="replace")
        if STUB_MARK in text and "NotImplementedError" in text:
            pending.append(rel)
    if pending:
        note("модулі ПР10", "ще не перенесено: " + ", ".join(pending))
    else:
        report(True, "модулі ПР10", "перенесено")


def check_stubs() -> None:
    """Які нові модулі ще заготовки. Це не помилка середовища — так і має
    бути на початку роботи."""
    pending = []
    for rel in NEW_STUBS:
        text = (HERE / rel).read_text(encoding="utf-8", errors="replace")
        if "NotImplementedError" in text:
            pending.append(rel)
    if pending:
        note("заготовки ПР11", "ще не реалізовано: " + ", ".join(pending))
    else:
        report(True, "заготовки ПР11", "усі модулі мають реалізацію")


def _usage_dict(usage) -> dict | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump(exclude_none=True)
    return dict(usage)


def check_call(provider: str) -> bool:
    """Пробний запит у режимі streaming: чи приходить `usage` в потоці і
    скільки коштує пробний запит за цінами з `.env`."""
    if provider not in ("openai", "ollama"):
        note("пробний запит", "пропущено: провайдер — mock")
        return True
    try:
        import openai
    except ImportError:
        report(False, "пробний запит", "перевірку пропущено: немає пакета openai")
        return False

    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY") or ("ollama" if provider == "ollama" else None)
    model = os.getenv("LLM_MODEL")
    if not (base_url and api_key and model):
        report(False, "пробний запит", "перевірку пропущено: налаштування неповні")
        return False

    timeout = float(os.getenv("LLM_TIMEOUT", "20"))
    client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)

    def run(with_usage: bool):
        kwargs = {"model": model, "messages": PROBE_MESSAGES, "temperature": 0.3,
                  "max_tokens": 200, "stream": True}
        if with_usage:
            kwargs["stream_options"] = {"include_usage": True}
        started = time.perf_counter()
        first = None
        text = []
        usage = None
        for chunk in client.chat.completions.create(**kwargs):
            if first is None:
                first = time.perf_counter() - started
            if chunk.choices and chunk.choices[0].delta.content:
                text.append(chunk.choices[0].delta.content)
            if getattr(chunk, "usage", None):
                usage = _usage_dict(chunk.usage)
        return "".join(text).strip(), usage, first, time.perf_counter() - started

    supports_usage = True
    try:
        try:
            answer, usage, first, total = run(with_usage=True)
        except openai.BadRequestError:
            supports_usage = False
            answer, usage, first, total = run(with_usage=False)
    except openai.APITimeoutError:
        report(False, "пробний запит", f"таймаут після {timeout:.0f} с")
        return False
    except openai.RateLimitError:
        report(False, "пробний запит", "429 — ліміт провайдера; спробуйте пізніше")
        return False
    except openai.AuthenticationError:
        report(False, "пробний запит", "401 — ключ невірний або протермінований")
        return False
    except openai.APIConnectionError as exc:
        report(False, "пробний запит", f"немає зʼєднання: {exc}")
        return False
    except openai.APIStatusError as exc:
        report(False, "пробний запит", f"статус {exc.status_code}: {exc}")
        return False
    except Exception as exc:  # noqa: BLE001 — діагностика, показуємо все
        report(False, "пробний запит", f"збій ({type(exc).__name__}): {exc}")
        return False

    if not answer:
        report(False, "пробний запит", "потік завершився без тексту")
        return False
    report(True, "пробний запит у режимі streaming",
           f"перший фрагмент через {first:.2f} с, разом {total:.2f} с")
    print(f"       {answer[:160]}{'…' if len(answer) > 160 else ''}")

    if not supports_usage:
        note("usage у потоці", "провайдер не прийняв stream_options — вартість у streaming "
                               "доведеться оцінювати за текстом і позначати як оцінку")
        return True
    if not usage:
        note("usage у потоці", "stream_options прийнято, але usage не прийшов — вартість "
                               "доведеться оцінювати за текстом")
        return True

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    price_in = float(os.getenv("LLM_PRICE_INPUT_PER_1M", "0.30"))
    price_out = float(os.getenv("LLM_PRICE_OUTPUT_PER_1M", "2.50"))
    cost = prompt_tokens / 1e6 * price_in + completion_tokens / 1e6 * price_out
    report(True, "usage у потоці",
           f"вхід {prompt_tokens}, вихід {completion_tokens}, кешовано провайдером {cached}; "
           f"поля: {', '.join(sorted(usage))}")
    note("вартість пробного запиту", f"{cost:.6f} $ за цінами з .env — від цього відштовхуйтеся, "
                                     "обираючи BUDGET_DAILY_USD")
    return True


def main() -> int:
    print("Перевірка середовища для ПР11\n")
    results = [check_python(), check_packages()]
    settings_ok, provider = check_settings()
    results += [settings_ok, check_dirs(), check_samples()]
    check_transfer()
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
