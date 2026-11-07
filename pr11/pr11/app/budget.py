"""Денний ліміт витрат: скільки застосунок може витратити за добу.

Обмеження частоти з ПР10 рахує запити за хвилину і не завадить
витратити місячний бюджет рівномірним потоком. Тут рахуються гроші за
день на весь застосунок. Ліній дві: мʼяка — попередження на частці
`WARN_AT` (оператор бачить, журнал отримує `WARNING`), жорстка —
відмова після вичерпання.

Модуль знає лише суму, ліміт і день. Шлюз (`app/gateway.py`) викликає
`check()` перед платним звертанням до провайдера і `record(cost)`
після нього; веб-рівень показує `status()` на сторінці.

Функції нижче — заготовки. Рішення з розділу 2 практичної роботи:

* де перевіряти: перед кожним платним викликом у шлюзі — обовʼязково
  (влучання в кеш безкоштовне й перевірки не потребує); чи ще й до
  запуску фонової роботи, щоб оператор дізнався одразу, а не подією
  через секунду;
* що з двома одночасними викликами на залишок бюджету: обидва пройдуть
  перевірку. Резервувати очікувану вартість наперед і уточнювати після
  — чи змиритися з невеликим перевищенням;
* стан на диску: лічильник у памʼяті обнулюється з перезапуском — і
  перезапуск «поповнює» бюджет. Файл у `state/` з датою і сумою,
  атомарний запис, як у ПР9;
* межа доби: за яким часом — локальним чи UTC; коли саме відновлюється
  ліміт і як це сказати операторові (`resets_at`);
* чернетка, яка вже генерується, коли бюджет вичерпався посеред неї:
  дати завершити чи перервати;
* вартість `None` (не визначено): рахувати як нуль, як оцінку, чи
  відмовляти, бо не знаємо, скільки витратили.
"""

import os
import json
import logging
import math
import tempfile
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

load_dotenv()

DAILY_USD = float(os.getenv("BUDGET_DAILY_USD", "1.00"))
WARN_AT = float(os.getenv("BUDGET_WARN_AT", "0.8"))
BUDGET_FILE = os.getenv("BUDGET_FILE", "state/budget.json")
if not math.isfinite(DAILY_USD) or DAILY_USD <= 0 or not 0 <= WARN_AT <= 1:
    raise ValueError("Денний бюджет має бути додатним, а WARN_AT — у межах від 0 до 1.")

logger = logging.getLogger("app.budget")
_lock = threading.RLock()
_RESERVATION_TTL = 3600


class BudgetExceeded(Exception):
    """Денний ліміт вичерпано. `spent`, `limit` — у доларах; `resets_at` —
    коли ліміт відновиться, у форматі ISO."""

    def __init__(self, spent: float, limit: float, resets_at: str) -> None:
        super().__init__(f"денний ліміт витрат {limit:.2f} $ вичерпано (витрачено {spent:.4f} $); "
                         f"відновиться {resets_at}")
        self.spent = spent
        self.limit = limit
        self.resets_at = resets_at


def settings() -> dict:
    """Чинні налаштування — для сторінки."""
    return {"daily_usd": DAILY_USD, "warn_at": WARN_AT, "file": BUDGET_FILE}


def _day() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _resets_at() -> str:
    tomorrow = datetime.now(timezone.utc).date() + timedelta(days=1)
    return datetime.combine(tomorrow, datetime.min.time(), timezone.utc).isoformat()


def _read() -> dict:
    today = _day()
    state = {"day": today, "spent": 0.0, "reservations": {}, "warning_emitted": False}
    if BUDGET_FILE and os.path.exists(BUDGET_FILE):
        try:
            with open(BUDGET_FILE, encoding="utf-8") as file:
                saved = json.load(file)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Не вдалося прочитати стан бюджету {BUDGET_FILE}: {exc}") from exc
        if not isinstance(saved, dict):
            raise RuntimeError(f"Файл бюджету {BUDGET_FILE} має некоректний формат.")
        if saved.get("day") == today:
            spent = saved.get("spent", 0)
            if isinstance(spent, bool) or not isinstance(spent, (int, float)) or not math.isfinite(spent) or spent < 0:
                raise RuntimeError(f"Файл бюджету {BUDGET_FILE} має некоректну суму.")
            reservations = saved.get("reservations", {})
            if not isinstance(reservations, dict):
                raise RuntimeError(f"Файл бюджету {BUDGET_FILE} має некоректні резерви.")
            state["spent"] = float(spent)
            state["warning_emitted"] = bool(saved.get("warning_emitted", False))
            now = time.time()
            state["reservations"] = {
                str(token): value for token, value in reservations.items()
                if isinstance(value, dict)
                and isinstance(value.get("amount"), (int, float))
                and not isinstance(value.get("amount"), bool)
                and math.isfinite(value["amount"])
                and value["amount"] >= 0
                and isinstance(value.get("created_at"), (int, float))
                and not isinstance(value.get("created_at"), bool)
                and math.isfinite(value["created_at"])
                and now - value["created_at"] < _RESERVATION_TTL
            }
    return state


def _write(state: dict) -> None:
    if not BUDGET_FILE:
        return
    path = os.path.abspath(BUDGET_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=os.path.dirname(path),
                                         prefix=".budget-", suffix=".tmp", delete=False) as file:
            temp_path = file.name
            json.dump(state, file, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def _reserved(state: dict) -> float:
    return sum(value["amount"] for value in state["reservations"].values())


def check(expected_cost: float | None = None, *, request_id: str | None = None,
          draft_id: str | None = None) -> str:
    """Переконатися, що платний виклик ще дозволений; інакше підняти
    `BudgetExceeded`. `expected_cost` — очікувана вартість, якщо ви
    вирішите резервувати її наперед."""
    amount = 0.0 if expected_cost is None else float(expected_cost)
    if not math.isfinite(amount) or amount < 0:
        raise ValueError("Очікувана вартість має бути скінченною невідʼємною сумою.")
    with _lock:
        state = _read()
        committed = state["spent"] + _reserved(state)
        if committed + amount > DAILY_USD:
            raise BudgetExceeded(committed, DAILY_USD, _resets_at())
        token = uuid.uuid4().hex
        state["reservations"][token] = {"amount": amount, "created_at": time.time()}
        projected = committed + amount
        if projected >= DAILY_USD * WARN_AT and not state["warning_emitted"]:
            state["warning_emitted"] = True
            logger.warning("budget.warning", extra={
                "request_id": request_id,
                "draft_id": draft_id,
                "spent": round(state["spent"], 8),
                "reserved": round(_reserved(state), 8),
                "limit": DAILY_USD,
                "resets_at": _resets_at(),
            })
        _write(state)
        return token


def record(cost: float | None, reservation_id: str | None = None) -> None:
    """Зарахувати вартість виконаного звертання до витрат дня."""
    if cost is not None and (not math.isfinite(cost) or cost < 0):
        raise ValueError("Вартість має бути скінченною невідʼємною сумою.")
    with _lock:
        state = _read()
        reservation = state["reservations"].pop(reservation_id, None) if reservation_id else None
        actual = float(cost) if cost is not None else (reservation["amount"] if reservation else 0.0)
        state["spent"] = round(state["spent"] + actual, 12)
        _write(state)


def status() -> dict:
    """Стан для `GET /api/usage` і сторінки: наприклад,
    `{"day": "2026-10-23", "spent": 0.1234, "limit": 1.0, "remaining": 0.8766,
    "warning": False, "exceeded": False, "resets_at": "2026-10-24T00:00:00+03:00"}`."""
    with _lock:
        state = _read()
        reserved = _reserved(state)
        projected = state["spent"] + reserved
        return {
            "day": state["day"],
            "spent": round(state["spent"], 8),
            "reserved": round(reserved, 8),
            "limit": DAILY_USD,
            "remaining": round(max(0.0, DAILY_USD - projected), 8),
            "warning": state["warning_emitted"] or projected >= DAILY_USD * WARN_AT,
            "exceeded": projected >= DAILY_USD,
            "resets_at": _resets_at(),
        }
