"""Перевірка середовища для ПР1.

Запуск:
    python check_env.py

Скрипт перевіряє три речі: версію Python, наявність потрібних пакетів і те,
що сервіси Open-Meteo доступні з вашої мережі. Це діагностика перед роботою,
а не зразок для наслідування: тут немає ані параметрів запиту, ані розбору
відповіді, ані обробки помилок — саме це ви проєктуєте самі (розділ 4 ПР1).
"""

import importlib
import sys

MIN_PYTHON = (3, 10)
PACKAGES = ("requests", "fastapi", "uvicorn")
HOSTS = (
    "https://geocoding-api.open-meteo.com/",
    "https://api.open-meteo.com/",
)
TIMEOUT_SECONDS = 10


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


def check_network() -> bool:
    try:
        import requests
    except ImportError:
        report(False, "доступ до мережі", "перевірку пропущено: немає пакета requests")
        return False

    ok = True
    for url in HOSTS:
        try:
            requests.get(url, timeout=TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            report(False, url, f"недоступний ({type(exc).__name__})")
            ok = False
        else:
            report(True, url, "")
    return ok


def report(ok: bool, what: str, hint: str) -> None:
    mark = "[ OK ]" if ok else "[ !! ]"
    tail = f" — {hint}" if hint else ""
    print(f"{mark} {what}{tail}")


def main() -> int:
    print("Перевірка середовища для ПР1\n")
    results = [check_python(), check_packages(), check_network()]
    print()
    if all(results):
        print("Середовище готове до роботи.")
        return 0
    print("Є проблеми — усуньте позначені [ !! ] і запустіть перевірку ще раз.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
