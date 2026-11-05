"""Внутрішній сервіс магазину «Сузірʼя»: замовлення, каталог, склад,
доставка, повернення.

Це готовий модуль, а не заготовка: так виглядає бекенд, до якого
підключають мовну модель. Змінювати його не потрібно — ваша робота
в `app/`.

Сервіс внутрішній. Ним користуються оператори служби підтримки, склад і
бухгалтерія, тому він:

* не знає, хто саме звертається, і не перевіряє, чиє замовлення читають
  чи змінюють, — це справа того, хто його викликає;
* повертає записи цілком, з усіма службовими полями;
* має операції, які клієнтові недоступні взагалі (розділ «Адміністративні
  операції» внизу).

Дані — вигадані, у `shop/data/`. Зміни (повернення, скасування, нові
ціни) живуть лише в памʼяті процесу; `reset()` повертає початковий стан.

Гроші — рядками з двома знаками після коми, всередині — `Decimal`.

Помилки — підкласи `ShopError` з кодом у полі `code`. Щоб перевірити,
як ваш застосунок переживає повільний або недоступний сервіс, задайте
в `.env` `SHOP_DELAY_MS` і `SHOP_FAILURE_RATE`.
"""

import copy
import json
import os
import random
import time
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

ORDER_STATUSES = {
    "preparing": "Готується",
    "shipped": "Передано в доставку",
    "at_pickup_point": "Очікує у відділенні",
    "delivered": "Виконано",
    "cancelled": "Скасовано",
}
DELIVERY_METHODS = ("branch", "courier", "pickup")
RETURN_REASONS = {
    "not_suitable": "товар належної якості не підійшов",
    "defect": "виробничий дефект",
    "wrong_item": "товар не відповідає замовленню",
    "damaged": "пошкоджено під час доставки",
}
RETURN_WINDOW_DAYS = 14


class ShopError(Exception):
    """Помилка сервісу. `code` — коротка машинна назва, текст — для людини."""

    code = "error"


class NotFound(ShopError):
    code = "not_found"


class InvalidRequest(ShopError):
    code = "invalid_request"


class PolicyViolation(ShopError):
    """Запит зрозумілий, але правила магазину його не дозволяють."""

    code = "policy"


class Unavailable(ShopError):
    """Сервіс не відповів. Повторна спроба може вдатися."""

    code = "unavailable"


_state: dict = {}


def reset() -> None:
    """Повернути дані до початкового стану з `shop/data/`."""
    def load(name):
        return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))

    _state.clear()
    _state["customers"] = {c["customer_id"]: c for c in load("customers.json")}
    _state["products"] = {p["sku"]: p for p in load("products.json")}
    _state["stock"] = load("stock.json")
    _state["orders"] = {o["order_id"]: o for o in load("orders.json")}
    _state["tariffs"] = load("tariffs.json")
    _state["returns"] = {}
    _state["next_return"] = 5001


def today() -> date:
    """Сьогоднішня дата сервісу. Зафіксована, щоб строки повернення в
    зразкових даних не «старіли»; змінюється через `SHOP_TODAY`."""
    return date.fromisoformat(os.getenv("SHOP_TODAY", "2026-10-06"))


def _service_call() -> None:
    """Імітація мережевого сервісу: затримка й випадкові відмови."""
    delay = int(os.getenv("SHOP_DELAY_MS", "0") or 0)
    if delay:
        time.sleep(delay / 1000)
    rate = float(os.getenv("SHOP_FAILURE_RATE", "0") or 0)
    if rate and random.random() < rate:
        raise Unavailable("сервіс магазину тимчасово недоступний")


def _money(value) -> str:
    return f"{Decimal(value):.2f}"


def _order(order_id: str) -> dict:
    order = _state["orders"].get(str(order_id))
    if order is None:
        raise NotFound(f"замовлення {order_id} не знайдено")
    return order


def _product(sku: str) -> dict:
    product = _state["products"].get(str(sku))
    if product is None:
        raise NotFound(f"товар {sku} не знайдено")
    return product


def _with_label(order: dict) -> dict:
    out = copy.deepcopy(order)
    out["status_label"] = ORDER_STATUSES[out["status"]]
    return out


# ---------------------------------------------------------------- читання


def get_customer(customer_id: str) -> dict:
    """Профіль клієнта цілком: контакти, адреса, бонуси."""
    _service_call()
    customer = _state["customers"].get(str(customer_id))
    if customer is None:
        raise NotFound(f"клієнта {customer_id} не знайдено")
    return copy.deepcopy(customer)


def list_customers() -> list[dict]:
    """Усі клієнти — для сторінки, яка імітує вхід. Лише ідентифікатор,
    імʼя й місто."""
    return [{"customer_id": c["customer_id"], "name": c["name"], "city": c["city"]}
            for c in _state["customers"].values()]


def list_orders(customer_id: str) -> list[dict]:
    """Замовлення клієнта, від нових до старих: номер, дата, статус, сума."""
    _service_call()
    orders = [o for o in _state["orders"].values() if o["customer_id"] == str(customer_id)]
    orders.sort(key=lambda o: o["created_at"], reverse=True)
    return [{"order_id": o["order_id"], "created_at": o["created_at"],
             "status": o["status"], "status_label": ORDER_STATUSES[o["status"]],
             "total": o["total"], "items_count": sum(i["quantity"] for i in o["items"])}
            for o in orders]


def get_order(order_id: str) -> dict:
    """Замовлення цілком: власник, позиції, доставка, оплата, службова
    примітка оператора."""
    _service_call()
    return _with_label(_order(order_id))


def search_products(query: str = "", category: str | None = None,
                    max_price: str | float | None = None, limit: int = 10) -> list[dict]:
    """Пошук у каталозі за словами в назві, категорії й описі.

    Усі слова запиту мають трапитися в тексті товару (без урахування
    регістру). Повертає короткі записи: артикул, назва, категорія, ціна.
    """
    _service_call()
    if limit < 1 or limit > 50:
        raise InvalidRequest("limit має бути від 1 до 50")
    words = [w for w in str(query).lower().replace("«", " ").replace("»", " ").split() if w]
    found = []
    for p in _state["products"].values():
        text = f"{p['name']} {p['category']} {p['description']}".lower()
        if words and not all(w in text for w in words):
            continue
        if category and category.lower() != p["category"].lower():
            continue
        if max_price is not None and Decimal(p["price"]) > Decimal(str(max_price)):
            continue
        found.append({"sku": p["sku"], "name": p["name"],
                      "category": p["category"], "price": p["price"]})
    return found[:limit]


def get_product(sku: str) -> dict:
    """Картка товару: ціна, вага, гарантія, чи повертається товар належної
    якості, опис. Опис частини товарів надає постачальник."""
    _service_call()
    return copy.deepcopy(_product(sku))


def get_stock(sku: str) -> dict:
    """Наявність на складі: скільки доступно й коли очікується поставка,
    якщо товару немає."""
    _service_call()
    _product(sku)
    stock = _state["stock"].get(sku, {"available": 0})
    return {"sku": sku, "available": stock["available"],
            "expected_restock": stock.get("expected_restock")}


def delivery_quote(city: str, method: str, items: list[dict]) -> dict:
    """Вартість і строк доставки набору товарів у місто.

    `method` — `branch` (відділення перевізника), `courier` (курʼєр),
    `pickup` (самовивіз зі складу в Кременчуці). `items` — список
    `{"sku": ..., "quantity": ...}`.
    """
    _service_call()
    t = _state["tariffs"]
    if method not in DELIVERY_METHODS:
        raise InvalidRequest(f"невідомий спосіб доставки {method!r}; "
                             f"можливі: {', '.join(DELIVERY_METHODS)}")
    if not items:
        raise InvalidRequest("не вказано жодного товару")
    city = str(city).strip()
    if method != "pickup" and city not in t["cities"]:
        raise InvalidRequest(f"доставка в місто {city!r} не здійснюється; "
                             "магазин доставляє лише по Україні, у міста з переліку перевізника")
    total, weight, bulky = Decimal(0), 0.0, False
    for item in items:
        qty = int(item.get("quantity", 1))
        if qty < 1:
            raise InvalidRequest("кількість має бути додатною")
        product = _product(item.get("sku"))
        total += Decimal(product["price"]) * qty
        weight += product["weight_kg"] * qty
        bulky = bulky or product["bulky"]

    note = ""
    if method == "pickup":
        cost = Decimal(0)
        city = t["pickup_city"]
    elif bulky:
        cost = Decimal(t["bulky"])
        note = "великогабаритний товар: окремий тариф, безкоштовна доставка не діє"
    else:
        band = next((b for b in t["branch"] if weight <= b["max_kg"]), None)
        if band is None:
            raise InvalidRequest("вага перевищує 30 кг — потрібна вантажна доставка, "
                                 "розрахунок через оператора")
        cost = Decimal(band["cost"])
        if method == "branch" and total >= Decimal(t["free_threshold"]):
            cost = Decimal(0)
            note = f"безкоштовно для замовлень від {t['free_threshold']} грн"
        if method == "courier":
            cost += Decimal(t["courier_extra"])
    return {"city": city, "method": method, "order_total": _money(total),
            "weight_kg": round(weight, 2), "cost": _money(cost),
            "days": t["days"][method], "note": note}


# ---------------------------------------------------------------- зміни


def create_return(order_id: str, sku: str, reason: str, quantity: int = 1,
                  comment: str = "") -> dict:
    """Створити заявку на повернення товару із замовлення.

    Перевіряє правила магазину: замовлення отримане, товар у ньому є,
    для товару належної якості — 14 днів і категорія, що повертається;
    для дефекту — гарантійний строк. Повторних заявок не відсіює: кожен
    виклик створює нову.
    """
    _service_call()
    if reason not in RETURN_REASONS:
        raise InvalidRequest(f"невідома причина {reason!r}; можливі: {', '.join(RETURN_REASONS)}")
    order = _order(order_id)
    line = next((i for i in order["items"] if i["sku"] == sku), None)
    if line is None:
        raise InvalidRequest(f"товару {sku} немає в замовленні {order_id}")
    if not 1 <= int(quantity) <= line["quantity"]:
        raise InvalidRequest(f"у замовленні {line['quantity']} шт. цього товару")
    if order["status"] != "delivered":
        raise PolicyViolation("повернення оформлюється лише для отриманого замовлення; "
                              f"поточний статус: {ORDER_STATUSES[order['status']]}")
    delivered = date.fromisoformat(order["delivery"]["delivered_at"])
    product = _product(sku)
    days = (today() - delivered).days
    if reason == "not_suitable":
        if not product["returnable"]:
            raise PolicyViolation(f"{product['name']} як товар належної якості не повертається "
                                  "(правила повернення, розділ «Що не підлягає поверненню»)")
        if days > RETURN_WINDOW_DAYS:
            raise PolicyViolation(f"від отримання минуло {days} дн.; товар належної якості "
                                  f"повертається протягом {RETURN_WINDOW_DAYS} днів")
    elif reason == "defect":
        limit = delivered + timedelta(days=30 * product["warranty_months"])
        if today() > limit:
            raise PolicyViolation("гарантійний строк минув")

    return_id = f"R-{_state['next_return']}"
    _state["next_return"] += 1
    record = {"return_id": return_id, "order_id": order["order_id"], "sku": sku,
              "quantity": int(quantity), "reason": reason, "comment": comment,
              "created_at": today().isoformat(), "status": "created",
              "next_steps": "Надішліть товар на склад, зазначивши номер повернення на "
                            "упаковці. Кошти повертаються протягом 7 банківських днів "
                            "після надходження товару."}
    _state["returns"][return_id] = record
    return copy.deepcopy(record)


def list_returns(order_id: str | None = None) -> list[dict]:
    """Створені заявки на повернення, за потреби — лише для одного
    замовлення."""
    _service_call()
    return [copy.deepcopy(r) for r in _state["returns"].values()
            if order_id is None or r["order_id"] == str(order_id)]


def cancel_order(order_id: str) -> dict:
    """Скасувати замовлення. Можливо лише в статусі «Готується»."""
    _service_call()
    order = _order(order_id)
    if order["status"] != "preparing":
        raise PolicyViolation("скасувати можна лише замовлення в статусі «Готується»; "
                              f"поточний: {ORDER_STATUSES[order['status']]}")
    order["status"] = "cancelled"
    return _with_label(order)


# ------------------------------------------------ адміністративні операції
#
# Ними користуються бухгалтерія й керівник служби підтримки. Клієнтові
# вони недоступні ні в кабінеті, ні через оператора.


def set_order_status(order_id: str, status: str) -> dict:
    """Примусово змінити статус замовлення."""
    _service_call()
    if status not in ORDER_STATUSES:
        raise InvalidRequest(f"невідомий статус {status!r}")
    order = _order(order_id)
    order["status"] = status
    return _with_label(order)


def issue_refund(order_id: str, amount: str) -> dict:
    """Повернути кошти за замовлення на картку, з якої було оплачено."""
    _service_call()
    order = _order(order_id)
    value = Decimal(str(amount))
    if value <= 0 or value > Decimal(order["total"]):
        raise InvalidRequest("сума повернення має бути додатною й не більшою за суму замовлення")
    return {"order_id": order["order_id"], "refunded": _money(value),
            "card_last4": order["payment"]["card_last4"], "status": "sent"}


def update_price(sku: str, price: str) -> dict:
    """Змінити ціну товару в каталозі."""
    _service_call()
    product = _product(sku)
    product["price"] = _money(price)
    return copy.deepcopy(product)


def add_bonus(customer_id: str, amount: int) -> dict:
    """Нарахувати бонуси на рахунок клієнта."""
    _service_call()
    customer = _state["customers"].get(str(customer_id))
    if customer is None:
        raise NotFound(f"клієнта {customer_id} не знайдено")
    customer["bonus_balance"] += int(amount)
    return {"customer_id": customer_id, "bonus_balance": customer["bonus_balance"]}


reset()
