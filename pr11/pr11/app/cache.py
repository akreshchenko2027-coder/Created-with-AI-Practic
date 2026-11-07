"""Кеш відповідей: однакове питання — одна оплата.

Повтори беруться звідусіль: «Повторити» без правок, пʼять натискань
поспіль, двоє операторів над одним зверненням. Кеш повертає готовий
результат, не звертаючись до провайдера, — і це видно операторові,
журналу й бюджету.

Модуль знає `Request` і `Result` з контракту провайдера і більше
нічого: ані хто питає, ані навіщо. Шлюз (`app/gateway.py`) перед
першою спробою питає `get(make_key(...))`, після вдалої генерації
кладе `put`; прапорець `fresh` у `Request.metadata` каже кеш оминути.

Функції нижче — заготовки. Рішення з розділу 2 практичної роботи:

* що входить у ключ: усе, що впливає на відповідь (провайдер, модель,
  версія prompt, повідомлення, температура, `max_tokens`), і нічого з
  того, що не впливає (клієнт, номер чернетки, час). Канонічне подання
  — `json.dumps(..., sort_keys=True)` → `hashlib.sha256`;
* термін життя: факти звернення змінюються; що таке «застаріло» для
  чернетки — година, день? Коли видаляти прострочене — при читанні чи
  окремо;
* сховище: словник у памʼяті (зникає з перезапуском) чи файл JSON у
  `state/` (переживає, але потребує обмеження розміру й атомарного
  запису). Що робити, коли записів більше за `MAX_ENTRIES`;
* що кешувати: результат запасного варіанту? підставного провайдера?
  відповідь, у якій `finish_reason` не `stop`;
* `fresh` оминає читання — а запис? Нова чернетка після «Повторити»
  має замінити стару в кеші чи ні;
* пʼять однакових запитів одночасно промахнуться всі — чи має другий
  чекати на перший (обʼєднання запитів у польоті) і скільки;
* лічильники для `stats`: влучання, промахи, обходи, записів зараз;
  чи скидати їх із перезапуском.
"""

import hashlib
import json
import math
import os
import tempfile
import threading
import time

from dotenv import load_dotenv

from .providers.base import Request, Result

load_dotenv()

ENABLED = os.getenv("CACHE_ENABLED", "1") == "1"
TTL_SECONDS = float(os.getenv("CACHE_TTL_SECONDS", "3600"))
MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "500"))
# Порожнє значення — кеш лише в памʼяті процесу.
CACHE_FILE = os.getenv("CACHE_FILE", "state/cache.json")
if not math.isfinite(TTL_SECONDS) or TTL_SECONDS <= 0 or MAX_ENTRIES < 1:
    raise ValueError("TTL кешу та максимальна кількість записів мають бути додатними.")

_lock = threading.RLock()
_entries: dict[str, dict] | None = None
_counts = {"hits": 0, "misses": 0, "bypassed": 0}


def settings() -> dict:
    """Чинні налаштування — для сторінки й журналу."""
    return {
        "enabled": ENABLED,
        "ttl_seconds": TTL_SECONDS,
        "max_entries": MAX_ENTRIES,
        "file": CACHE_FILE or None,
    }


def make_key(request: Request, prompt_version: str) -> str:
    """Ключ кешу для запиту: однакові запити — однаковий ключ, будь-яка
    відмінність у тому, що впливає на відповідь, — інший."""
    if not request.metadata.get("provider"):
        from . import gateway

        provider = gateway.PROVIDER
    else:
        provider = request.metadata["provider"]
    value = {
        "provider": provider,
        "model": request.model,
        "prompt_version": prompt_version,
        "messages": request.messages,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    }
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load() -> dict[str, dict]:
    global _entries
    if _entries is not None:
        return _entries
    entries: dict[str, dict] = {}
    if CACHE_FILE and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, encoding="utf-8") as file:
                stored = json.load(file)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Не вдалося прочитати кеш {CACHE_FILE}: {exc}") from exc
        if not isinstance(stored, dict) or not isinstance(stored.get("entries"), dict):
            raise RuntimeError(f"Файл кешу {CACHE_FILE} має некоректний формат.")
        for key, entry in stored["entries"].items():
            if not isinstance(entry, dict) or not isinstance(entry.get("expires_at"), (int, float)):
                continue
            if not math.isfinite(entry["expires_at"]):
                continue
            result = entry.get("result")
            if not isinstance(result, dict):
                continue
            try:
                entries[str(key)] = {
                    "expires_at": float(entry["expires_at"]),
                    "result": Result(**result),
                }
            except TypeError:
                continue
    _entries = entries
    _purge()
    return _entries


def _persist() -> None:
    if not CACHE_FILE:
        return
    path = os.path.abspath(CACHE_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "entries": {
            key: {
                "expires_at": entry["expires_at"],
                "result": entry["result"].__dict__,
            }
            for key, entry in (_entries or {}).items()
        }
    }
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=os.path.dirname(path),
                                         prefix=".cache-", suffix=".tmp", delete=False) as file:
            temp_path = file.name
            json.dump(payload, file, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def _purge() -> bool:
    entries = _entries or {}
    now = time.time()
    expired = [key for key, entry in entries.items() if entry["expires_at"] <= now]
    for key in expired:
        entries.pop(key, None)
    return bool(expired)


def get(key: str) -> Result | None:
    """Результат за ключем, якщо він є й не прострочений; інакше `None`.
    Коли кеш вимкнено — завжди `None`."""
    with _lock:
        if not ENABLED:
            _counts["misses"] += 1
            return None
        entries = _load()
        if _purge():
            _persist()
        entry = entries.get(key)
        if entry is None:
            _counts["misses"] += 1
            return None
        _counts["hits"] += 1
        entries.pop(key)
        entries[key] = entry
        _persist()
        return entry["result"]


def put(key: str, result: Result) -> None:
    """Зберегти результат. Коли кеш вимкнено — нічого не робити."""
    if not ENABLED:
        return
    with _lock:
        entries = _load()
        _purge()
        entries.pop(key, None)
        entries[key] = {
            "expires_at": time.time() + TTL_SECONDS,
            "result": result,
        }
        while len(entries) > MAX_ENTRIES:
            entries.pop(next(iter(entries)))
        _persist()


def note_bypassed() -> None:
    with _lock:
        _counts["bypassed"] += 1


def stats() -> dict:
    """Лічильники для `GET /api/usage` і сторінки: наприклад,
    `{"hits": 3, "misses": 7, "bypassed": 1, "entries": 6}`."""
    with _lock:
        entries = _load()
        if _purge():
            _persist()
        return {**_counts, "entries": len(entries), "enabled": ENABLED}
