"""Адаптер до помічника клієнта з ПР8: відповідь із викликом інструментів.

Застосунок імпортується з папки `pr8` як є (`AGENT_APP_DIR` у `.env`,
типово — сусідня папка репозиторію). Приклад — це питання клієнта і
клієнт сеансу: `assistant.answer(question, customer_id)`. Результат —
текст і **журнал викликів інструментів**: саме він потрібен перевіркам
на недозволені виклики, аргументи й витоки — відповідь сама по собі не
каже, що модель попросила і що з цим зробив код.

Що успадковується від ПР8:

* налаштування — з `pr8/.env` (модель, `TOOL_MAX_ROUNDS`, `SHOP_TODAY`,
  `SHOP_FAILURE_RATE`); змінна з `pr12/.env` із тим самим іменем має
  перевагу;
* перед кожним прикладом дані магазину скидаються (`service.reset()`),
  інакше заявка, створена одним прикладом, змінить відповідь наступному;
* кешу й бюджету в ПР8 не було — кожен приклад платний.

Підлаштувати, якщо потрібно: `_system_prompt()` — адаптер не знає, де у
вашому `pr8/app/llm.py` лежить інструкція; без неї перевірка
`no_system_prompt` працюватиме лише за маркерами з прикладу. Якщо ви
змінили склад `Answer` чи `ToolTrace`, поправте `run()`.
"""

import dataclasses
import os
import sys
import time
import types
from pathlib import Path

NAME = "agent"

_default_dir = Path(__file__).resolve().parents[4] / "pr8" / "pr8"
APP_DIR = Path(os.getenv("AGENT_APP_DIR") or _default_dir).resolve()

_app: types.SimpleNamespace | None = None


def setup() -> dict:
    """Імпортувати помічник ПР8 і сервіс магазину; повернути конфігурацію."""
    global _app
    if not (APP_DIR / "app" / "assistant.py").exists():
        raise SystemExit(f"не знайдено помічник ПР8 у {APP_DIR}: "
                         "задайте AGENT_APP_DIR у .env (папка, у якій лежать app/ і shop/)")
    sys.path.insert(0, str(APP_DIR))
    from app import assistant, tools  # noqa: E402 — після sys.path
    from shop import service  # noqa: E402
    _app = types.SimpleNamespace(assistant=assistant, service=service, tools=tools)
    return {
        "provider": "openai",
        "model": os.getenv("LLM_MODEL"),
        "temperature": os.getenv("LLM_TEMPERATURE"),
        "max_rounds": getattr(assistant, "MAX_ROUNDS", None),
        "shop_today": os.getenv("SHOP_TODAY"),
        "app_dir": str(APP_DIR),
    }


def before_case(case: dict) -> None:
    """Повернути дані магазину до початкового стану."""
    _app.service.reset()


def _system_prompt() -> str | None:
    """Текст системної інструкції помічника, якщо його можна дістати.

    Поверніть тут свою інструкцію з `pr8/app/llm.py` — наприклад,
    `from app import llm; return llm.SYSTEM_PROMPT`. Нижче — спроба
    вгадати типові назви; якщо жодна не підходить, повертається None.
    """
    try:
        from app import llm  # noqa: E402
    except ImportError:
        return None
    for name in ("SYSTEM_PROMPT", "SYSTEM", "INSTRUCTION", "INSTRUCTIONS", "SYSTEM_MESSAGE"):
        value = getattr(llm, name, None)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _as_dict(obj) -> dict:
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return dict(obj)
    return dict(vars(obj))


def run(case: dict) -> dict:
    """Виконати один приклад: питання клієнта від імені клієнта сеансу."""
    question = case["input"]["question"]
    customer_id = case["input"]["customer_id"]
    started = time.perf_counter()
    try:
        answer = _app.assistant.answer(question, customer_id)
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001 — контрольована відмова помічника
        kind = getattr(exc, "kind", None) or type(exc).__name__
        return {"text": None, "status": "error", "error": {"kind": kind, "message": str(exc)},
                "calls": [], "system_prompt": _system_prompt(),
                "elapsed": round(time.perf_counter() - started, 3)}

    calls = [_as_dict(c) for c in (getattr(answer, "calls", None) or [])]
    tool_schemas = {
        item["function"]["name"]: item["function"]["parameters"]
        for item in _app.tools.specs()
    }
    elapsed = getattr(answer, "elapsed", None)
    return {
        "text": getattr(answer, "text", None),
        "status": "ok",
        "error": None,
        "calls": calls,
        "tool_schemas": tool_schemas,
        "system_prompt": _system_prompt(),
        "provider": "openai",
        "model": getattr(answer, "model", None),
        "usage": getattr(answer, "usage", None),
        "cost": None,
        "rounds": getattr(answer, "rounds", None),
        "stopped": getattr(answer, "stopped", None),
        "elapsed_by_kind": elapsed if isinstance(elapsed, dict) else None,
        "elapsed": round(time.perf_counter() - started, 3),
    }
