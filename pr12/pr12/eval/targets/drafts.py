"""Адаптер до помічника оператора з ПР11: чернетка відповіді клієнтові.

Застосунок імпортується з папки `pr11` як є (`DRAFTS_APP_DIR` у `.env`,
типово — сусідня папка репозиторію); нічого не копіюється. Приклад
проходить тим самим шляхом, що й замовлення зі сторінки, але без
веб-рівня й фонової задачі: `drafting.build_request(sample)` →
`gateway.generate(...)` → `drafting.check(text, sample)`. Так один
приклад — один виклик, а перевірки бачать і запит (системну
інструкцію), і результат (текст, модель, токени, вартість).

Що успадковується від ПР11 і що це означає для прогону:

* налаштування — з `pr11/.env` (провайдер, модель, температура,
  таймаути, ціни, кеш, бюджет); змінна з `pr12/.env` із тим самим
  іменем має перевагу, бо завантажується першою;
* кеш оминається (`metadata["fresh"] = True`): інакше k прогонів одного
  прикладу повернуть одну відповідь із кешу, і стабільності не виміряти.
  `EVAL_USE_CACHE=1` у `.env` вмикає кеш назад;
* бюджет дня діє: прогін набору — платні виклики, і вони враховуються
  так само, як замовлення операторів. Вичерпаний бюджет — `status:
  error`, вид `budget`, приклад не пройде;
* журнал пишеться у `logs/` поточної папки (`pr12/logs/app.jsonl`),
  якщо `LOG_FILE` відносний. Записи прогону там можна відрізнити за
  `metadata["eval"]`, якщо ваш шлюз кладе метадані в журнал.

Підлаштувати, якщо потрібно: сигнатуру `gateway.generate` — у ПР10 її
дозволено було змінити (генератор, черга замість зворотних викликів).
Це єдине місце, яке її знає; решта конвеєра працює зі словником
результату.
"""

import logging
import os
import sys
import time
import types
from pathlib import Path

NAME = "drafts"

_default_dir = Path(__file__).resolve().parents[4] / "pr11" / "pr11"
APP_DIR = Path(os.getenv("DRAFTS_APP_DIR") or _default_dir).resolve()
USE_CACHE = os.getenv("EVAL_USE_CACHE", "0") == "1"

_app: types.SimpleNamespace | None = None


def setup() -> dict:
    """Імпортувати застосунок ПР11 і повернути його чинну конфігурацію."""
    global _app
    if not (APP_DIR / "app" / "drafting.py").exists():
        raise SystemExit(f"не знайдено застосунок ПР11 у {APP_DIR}: "
                         "задайте DRAFTS_APP_DIR у .env (папка, у якій лежить app/)")
    sys.path.insert(0, str(APP_DIR))
    from app import drafting, gateway, samples  # noqa: E402 — після sys.path
    from app.providers.base import ProviderError  # noqa: E402
    try:
        from app import telemetry  # noqa: E402
        telemetry.setup()
        # Журнал — у файл; консольний обробник ПР11 знімаємо, щоб записи
        # шлюзу не перемішувалися з таблицею прогону.
        root = logging.getLogger()
        for handler in list(root.handlers):
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                root.removeHandler(handler)
    except Exception:  # noqa: BLE001 — журнал не є умовою прогону
        pass
    _app = types.SimpleNamespace(drafting=drafting, gateway=gateway, samples=samples,
                                 ProviderError=ProviderError)

    config: dict = {}
    try:
        config.update(gateway.settings())
    except Exception:  # noqa: BLE001
        pass
    config["prompt_version"] = getattr(drafting, "PROMPT_VERSION", None)
    config["temperature"] = getattr(gateway, "TEMPERATURE", None)
    config["max_tokens"] = getattr(gateway, "MAX_TOKENS", None)
    config["app_dir"] = str(APP_DIR)
    config["use_cache"] = USE_CACHE
    return config


def before_case(case: dict) -> None:
    """Зареєструвати приклад серед зразків: підставний провайдер шукає
    `reply_example` за `metadata["request_id"]` у `app.samples`, а
    прикладів із набору там немає."""
    sample = case["input"]
    registry = getattr(_app.samples, "_by_id", None)
    if isinstance(registry, dict) and sample.get("id"):
        registry[sample["id"]] = sample


def _error(kind: str, message: str, started: float, **extra) -> dict:
    return {"text": None, "status": "error", "error": {"kind": kind, "message": message},
            "calls": [], "elapsed": round(time.perf_counter() - started, 3), **extra}


def run(case: dict) -> dict:
    """Виконати один приклад: скласти запит, згенерувати, перевірити
    чернетку кодом застосунку."""
    sample = dict(case["input"])
    started = time.perf_counter()

    try:
        request = _app.drafting.build_request(sample)
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001 — контрольована відмова на вході
        return _error(type(exc).__name__, str(exc), started)

    metadata = dict(getattr(request, "metadata", {}) or {})
    metadata.setdefault("request_id", sample.get("id"))
    metadata["fresh"] = not USE_CACHE
    metadata["eval"] = True
    request.metadata = metadata

    system_prompt = next((m.get("content") for m in request.messages
                          if isinstance(m, dict) and m.get("role") == "system"), None)
    pieces: list[str] = []
    statuses: list[dict] = []
    try:
        result = _app.gateway.generate(request, pieces.append, statuses.append, lambda: False)
    except _app.ProviderError as exc:
        return _error(exc.kind, exc.message, started, system_prompt=system_prompt,
                      statuses=statuses)
    except TypeError as exc:
        raise SystemExit(f"gateway.generate викликано не так, як очікує ваш шлюз ({exc}); "
                         "підлаштуйте run() в eval/targets/drafts.py") from exc

    text = getattr(result, "text", None) or "".join(pieces)
    try:
        warnings = list(_app.drafting.check(text, sample) or [])
    except NotImplementedError:
        warnings = []

    return {
        "text": text,
        "status": "ok",
        "error": None,
        "calls": [],
        "system_prompt": system_prompt,
        "provider": getattr(result, "provider", None),
        "model": getattr(result, "model", None),
        "usage": getattr(result, "usage", None),
        "cost": getattr(result, "cost", None),
        "cost_estimated": getattr(result, "cost_estimated", False),
        "cached": getattr(result, "cached", False),
        "fallback": any(s.get("fallback") for s in statuses if isinstance(s, dict)),
        "attempts": max((s.get("attempt", 0) for s in statuses if isinstance(s, dict)), default=None),
        "warnings": warnings,
        "elapsed": round(time.perf_counter() - started, 3),
    }
