"""Перевірка середовища для ПР7.

Запуск:
    python check_env.py

Скрипт перевіряє версію Python, наявність потрібних пакетів, налаштування
в `.env`, наявність зразків і довідників та те, що модель справді приймає
зображення: малює невелику картинку з числом, надсилає її моделі й
показує відповідь, час і витрачені токени.

Запустіть перевірку заздалегідь: якщо локальна Ollama недоступна або
обрана модель не приймає зображень, зʼясувати це краще до заняття.

Це діагностика перед роботою, а не зразок для наслідування: тут немає
ані підготовки зображення, ані схеми, ані перевірки відповіді — саме це
ви проєктуєте самі.
"""

import base64
import importlib
import io
import json
import os
import sys
import time
from pathlib import Path

MIN_PYTHON = (3, 10)
PACKAGES = ("openai", "dotenv", "fastapi", "uvicorn", "pydantic", "PIL", "multipart")
OPTIONAL_PACKAGES = ("jsonschema",)
SETTINGS = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")
OPTIONAL_SETTINGS = ("LLM_MAX_TOKENS", "LLM_TIMEOUT", "UPLOAD_MAX_MB", "IMAGE_MAX_SIDE")
HERE = Path(__file__).parent
SAMPLES = HERE / "samples"
REFERENCE = HERE / "reference"

HINTS = {
    "LLM_BASE_URL": "не задано: скопіюйте .env.example у .env",
    "LLM_API_KEY": "не задано: для локальної Ollama вкажіть технічне значення ollama",
    "LLM_MODEL": "не задано: завантажте мультимодальну модель, наприклад gemma3:4b",
}

# Пробне зображення: одне число, намальоване вбудованим шрифтом Pillow.
# Перевіряє лише, що модель приймає зображення й читає з нього цифри.
PROBE_NUMBER = "4719"


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
            note(f"пакет {name}", "не встановлено; потрібен лише для схеми, написаної вручну")
        else:
            report(True, f"пакет {name}", "")
    return ok


def check_settings() -> bool:
    """Перевірити локальні налаштування моделі. Значення ключа не друкуємо."""
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


def check_samples() -> bool:
    ok = True
    for sub in ("clean", "degraded"):
        files = list((SAMPLES / sub).glob("*.*"))
        good = len(files) >= 5
        report(good, f"зразки samples/{sub}/: {len(files)} файлів",
               "" if good else "замало — перевірте, чи повністю розпаковано архів")
        ok = ok and good
    for path in (SAMPLES / "expected.json", REFERENCE / "company.json",
                 REFERENCE / "suppliers.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            report(False, str(path.relative_to(HERE)), f"не читається: {exc}")
            ok = False
        else:
            report(True, str(path.relative_to(HERE)), "")
    return ok


def probe_image() -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (360, 120), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=64)
    except TypeError:          # Pillow старший за 10.1 не знає size
        font = ImageFont.load_default()
    draw.text((30, 25), f"No {PROBE_NUMBER}", font=font, fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def check_call() -> bool:
    """Надіслати пробне зображення й зміряти час."""
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

    try:
        data = base64.b64encode(probe_image()).decode("ascii")
    except Exception as exc:  # noqa: BLE001 — діагностика, показуємо все
        report(False, "пробне зображення", f"не вдалося намалювати ({type(exc).__name__}): {exc}")
        return False

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=60)
    started = time.perf_counter()
    try:
        answer = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Яке число написано на зображенні? Лише цифри."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
            ]}],
            max_tokens=20,
            temperature=0,
        )
    except Exception as exc:  # noqa: BLE001 — діагностика, показуємо все
        report(False, "пробний запит із зображенням", f"збій ({type(exc).__name__}): {exc}")
        return False
    elapsed = time.perf_counter() - started

    text = (answer.choices[0].message.content or "").strip()
    report(True, "пробний запит із зображенням", f"відповідь за {elapsed:.2f} с: {text[:60]!r}")
    if PROBE_NUMBER not in text:
        note("читання", f"модель не назвала {PROBE_NUMBER} — подивіться, що саме вона повернула")
    usage = getattr(answer, "usage", None)
    if usage:
        print(f"       токенів: запит {usage.prompt_tokens} (з них більшість — зображення "
              f"360 x 120), відповідь {usage.completion_tokens}")
    return True


def main() -> int:
    print("Перевірка середовища для ПР7\n")
    results = [check_python(), check_packages(), check_settings(), check_samples(),
               check_call()]
    print()
    if all(results):
        print("Середовище готове до роботи.")
        return 0
    print("Є проблеми — усуньте позначені [ !! ] і запустіть перевірку ще раз.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
