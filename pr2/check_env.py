"""Перевірка середовища для ПР2.

Запуск:
    python check_env.py

Скрипт перевіряє версію Python, наявність потрібних пакетів і те, що ваги
моделі доступні. Останнє важливо зробити заздалегідь: ваги завантажуються
з мережі при першому використанні, і робити це вперше під час заняття —
погана ідея.

Це діагностика перед роботою, а не зразок для наслідування: тут немає ані
порога впевненості, ані розбору результату, ані обробки помилок — саме це
ви проєктуєте самі.
"""

import importlib
import sys
import time

MIN_PYTHON = (3, 10)
PACKAGES = ("ultralytics", "fastapi", "uvicorn", "multipart")
WEIGHTS = "yolov8n.pt"


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


def check_weights() -> bool:
    try:
        from ultralytics import YOLO
    except ImportError:
        report(False, f"ваги {WEIGHTS}", "перевірку пропущено: немає пакета ultralytics")
        return False

    print(f"       завантаження {WEIGHTS} — перший раз може тривати хвилину…")
    started = time.perf_counter()
    try:
        YOLO(WEIGHTS)
    except Exception as exc:
        report(False, f"ваги {WEIGHTS}", f"не вдалося завантажити ({type(exc).__name__})")
        return False
    elapsed = time.perf_counter() - started
    report(True, f"ваги {WEIGHTS}", f"готові за {elapsed:.1f} с")
    return True


def report(ok: bool, what: str, hint: str) -> None:
    mark = "[ OK ]" if ok else "[ !! ]"
    tail = f" — {hint}" if hint else ""
    print(f"{mark} {what}{tail}")


def main() -> int:
    print("Перевірка середовища для ПР2\n")
    results = [check_python(), check_packages(), check_weights()]
    print()
    if all(results):
        print("Середовище готове до роботи.")
        return 0
    print("Є проблеми — усуньте позначені [ !! ] і запустіть перевірку ще раз.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
