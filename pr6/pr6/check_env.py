"""Перевірка середовища для ПР6.

Запуск:
    python check_env.py

Скрипт перевіряє версію Python, наявність потрібних пакетів, налаштування
в `.env`, наявність колекції документів, доступність моделі ембедінгів і
те, що мовна модель справді відповідає: надсилає пробний запит з одним
речення контексту й питанням до нього та показує час і витрачені токени.

Запустіть перевірку заздалегідь: якщо ключ не працює або сервіс
недоступний, зʼясувати це краще до заняття.

Це діагностика перед роботою, а не зразок для наслідування: тут немає
ані відбору фрагментів, ані інструкції з обмеженнями, ані перевірки
відповіді — саме це ви проєктуєте самі.
"""

import importlib
import os
import sys
import time
from pathlib import Path

MIN_PYTHON = (3, 10)
PACKAGES = ("openai", "dotenv", "fastapi", "uvicorn", "pydantic",
            "sentence_transformers", "numpy")
OPTIONAL_PACKAGES = ("rank_bm25",)
SETTINGS = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")
OPTIONAL_SETTINGS = ("EMBEDDING_MODEL", "SEARCH_TOP_K", "SIMILARITY_THRESHOLD",
                     "RAG_CONTEXT_CHUNKS", "RAG_CONTEXT_BUDGET")
DEFAULT_MODEL = "intfloat/multilingual-e5-small"
DOCS_DIR = Path(__file__).parent / "docs"
MIN_DOCS = 10

HINTS = {
    "LLM_BASE_URL": "не задано: скопіюйте .env.example у .env",
    "LLM_API_KEY": "не задано: перенесіть ключ із .env попередніх робіт або візьміть "
                   "у Google AI Studio (aistudio.google.com)",
    "LLM_MODEL": "не задано: назву моделі дивіться в Google AI Studio",
}

# Пробний запит: одне речення «контексту» і питання до нього. Перевіряє
# лише, що модель відповідає і бере факт із наданого тексту. Якою має бути
# ваша інструкція і ваш контекст — це не показує.
PROBE_CONTEXT = "Посилка зберігається у відділенні перевізника 7 днів."
PROBE_QUESTION = "Скільки днів посилка чекає у відділенні?"
PROBE_PROMPT = (f"Відповідай одним коротким реченням лише на підставі тексту.\n\n"
                f"Текст: {PROBE_CONTEXT}\n\nПитання: {PROBE_QUESTION}")


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
            note(f"пакет {name}", "не встановлено; потрібен лише за вашим рішенням "
                                  "(пошук за словами / схема вручну)")
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


def check_docs() -> bool:
    files = [p for p in DOCS_DIR.glob("*.md") if p.name.lower() != "readme.md"]
    ok = len(files) >= MIN_DOCS
    size = sum(p.stat().st_size for p in files)
    report(ok, f"колекція docs/: {len(files)} документів, {size // 1024} КБ",
           "" if ok else "документів замало — перевірте, чи повністю розпаковано архів")
    return ok


def check_embeddings() -> bool:
    """Завантажити модель ембедінгів і закодувати одне речення."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        report(False, "модель ембедінгів", "перевірку пропущено: немає пакета sentence-transformers")
        return False

    model_name = os.getenv("EMBEDDING_MODEL") or DEFAULT_MODEL
    started = time.perf_counter()
    try:
        model = SentenceTransformer(model_name)
        vector = model.encode(["query: " + PROBE_QUESTION], normalize_embeddings=True)
    except Exception as exc:  # noqa: BLE001 — діагностика, показуємо все
        report(False, f"модель ембедінгів {model_name}",
               f"збій ({type(exc).__name__}): {exc}")
        return False
    elapsed = time.perf_counter() - started
    report(True, f"модель ембедінгів {model_name}",
           f"готова за {elapsed:.1f} с, розмірність {vector.shape[1]}")
    return True


def check_call() -> bool:
    """Надіслати пробний запит із контекстом і зміряти час."""
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
            messages=[{"role": "user", "content": PROBE_PROMPT}],
            max_tokens=60,
            temperature=0,
        )
    except Exception as exc:  # noqa: BLE001 — діагностика, показуємо все
        report(False, "пробний запит", f"збій ({type(exc).__name__}): {exc}")
        return False
    elapsed = time.perf_counter() - started

    text = (answer.choices[0].message.content or "").strip()
    report(True, "пробний запит", f"відповідь за {elapsed:.2f} с: {text[:100]!r}")
    if "7" not in text:
        note("ґрунтованість", "у відповіді немає числа з контексту — подивіться, "
                              "що саме повернула модель")
    usage = getattr(answer, "usage", None)
    if usage:
        print(f"       токенів: запит {usage.prompt_tokens}, "
              f"відповідь {usage.completion_tokens}, разом {usage.total_tokens}")
    return True


def main() -> int:
    print("Перевірка середовища для ПР6\n")
    results = [check_python(), check_packages(), check_settings(), check_docs(),
               check_embeddings(), check_call()]
    print()
    if all(results):
        print("Середовище готове до роботи.")
        return 0
    print("Є проблеми — усуньте позначені [ !! ] і запустіть перевірку ще раз.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
