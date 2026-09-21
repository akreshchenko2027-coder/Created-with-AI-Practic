"""Перевірка середовища для ПР5.

Запуск:
    python check_env.py

Скрипт перевіряє версію Python, наявність потрібних пакетів, налаштування
в `.env`, наявність колекції документів і те, що модель ембедінгів справді
працює: завантажує її, кодує три речення й показує, наскільки схожими
вона їх вважає.

Модель завантажується з мережі під час першого запуску — це кілька
сотень мегабайт. Запустіть перевірку заздалегідь: робити це вперше під
час заняття — погана ідея.

Це діагностика перед роботою, а не зразок для наслідування: тут немає
ані поділу на фрагменти, ані індексу, ані пошуку — саме це ви
проєктуєте самі.
"""

import importlib
import os
import sys
import time
from pathlib import Path

MIN_PYTHON = (3, 10)
PACKAGES = ("sentence_transformers", "numpy", "rank_bm25", "fastapi", "uvicorn", "dotenv")
OPTIONAL_SETTINGS = ("EMBEDDING_MODEL", "SEARCH_TOP_K", "SIMILARITY_THRESHOLD",
                     "CHUNK_SIZE", "CHUNK_OVERLAP")
DEFAULT_MODEL = "intfloat/multilingual-e5-small"
DOCS_DIR = Path(__file__).parent / "docs"
MIN_DOCS = 10

# Три речення для пробного кодування: перше й друге — про одне й те саме
# різними словами, третє — про інше. Модель має «побачити» це в числах.
PROBE = (
    "Як повернути гроші за товар?",
    "Кошти повертаються на рахунок, з якого була оплата, протягом 7 банківських днів.",
    "Навушники не заряджаються в кейсі.",
)


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


def check_settings() -> str:
    """Прочитати .env, якщо він є. Повертає назву моделі."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        note("налаштування .env", "перевірку пропущено: немає пакета python-dotenv")
        return DEFAULT_MODEL

    if not Path(".env").exists():
        note("файл .env", "немає; діятимуть значення за замовчуванням. "
                          "Скопіюйте .env.example у .env, щоб їх змінювати")
    load_dotenv()
    for name in OPTIONAL_SETTINGS:
        value = os.getenv(name)
        if value:
            report(True, f"налаштування {name}", value)
        else:
            note(f"налаштування {name}", "не задано; діє значення за замовчуванням із коду")
    return os.getenv("EMBEDDING_MODEL") or DEFAULT_MODEL


def check_docs() -> bool:
    files = [p for p in DOCS_DIR.glob("*.md") if p.name.lower() != "readme.md"]
    ok = len(files) >= MIN_DOCS
    size = sum(p.stat().st_size for p in files)
    report(ok, f"колекція docs/: {len(files)} документів, {size // 1024} КБ",
           "" if ok else "документів замало — перевірте, чи повністю розпаковано архів")
    return ok


def check_model(model_name: str) -> bool:
    """Завантажити модель і закодувати пробні речення."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        report(False, "модель ембедінгів", "перевірку пропущено: немає пакета sentence-transformers")
        return False

    print(f"       завантаження {model_name} — перший раз може тривати кілька хвилин…")
    started = time.perf_counter()
    try:
        model = SentenceTransformer(model_name)
    except Exception as exc:  # noqa: BLE001 — діагностика, показуємо все
        report(False, f"модель {model_name}", f"не завантажилась ({type(exc).__name__}): {exc}")
        return False
    loaded = time.perf_counter() - started
    max_len = getattr(model, "max_seq_length", None)
    report(True, f"модель {model_name}",
           f"завантажено за {loaded:.1f} с, максимальна довжина входу {max_len} токенів")

    # Моделі сімейства e5 очікують префікси «query: » і «passage: » —
    # так написано в картці моделі. Іншій моделі вони можуть бути
    # не потрібні; що потрібно вашій — читайте в її картці.
    prefix = "query: " if "e5" in model_name.lower() else ""
    started = time.perf_counter()
    try:
        vectors = model.encode([prefix + s for s in PROBE], normalize_embeddings=True)
    except Exception as exc:  # noqa: BLE001
        report(False, "пробне кодування", f"збій ({type(exc).__name__}): {exc}")
        return False
    elapsed = time.perf_counter() - started

    dim = vectors.shape[1]
    sim_close = float(vectors[0] @ vectors[1])
    sim_far = float(vectors[0] @ vectors[2])
    report(True, "пробне кодування",
           f"3 речення за {elapsed * 1000:.0f} мс, розмірність {dim}")
    print(f"       схожість «{PROBE[0]}»")
    print(f"         з «{PROBE[1][:48]}…»: {sim_close:.3f}")
    print(f"         з «{PROBE[2]}»: {sim_far:.3f}")
    if sim_close <= sim_far:
        note("схожість", "перефразування не ближче за стороннє речення — "
                         "перевірте модель і префікси")
    return True


def main() -> int:
    print("Перевірка середовища для ПР5\n")
    results = [check_python(), check_packages()]
    model_name = check_settings()
    results.append(check_docs())
    results.append(check_model(model_name))
    print()
    if all(results):
        print("Середовище готове до роботи.")
        return 0
    print("Є проблеми — усуньте позначені [ !! ] і запустіть перевірку ще раз.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
