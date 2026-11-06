import datetime
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterator

from . import drafting, gateway
from .providers.base import ProviderError

TERMINAL = {"done", "failed", "cancelled"}
MAX_DRAFTS = 200
logger = logging.getLogger(__name__)


@dataclass
class Draft:
    """Стан однієї чернетки — те, що зберігається в реєстрі і йде на
    сторінку як знімок.

    `status` — `queued`, `running`, `done`, `failed`, `cancelled`.
    `text` — накопичений текст. `attempts`, `provider`, `model`,
    `fallback` — звідки взявся результат. `error` — остаточна помилка,
    якщо є: `kind`, `message`, `retryable`. `elapsed` — секунди, наприклад
    `{"first_token": 0.8, "total": 4.1}`. `warnings` — зауваження до
    чернетки з `drafting.check`. `events` — журнал подій для потоку;
    у знімок на сторінку не входить.
    """

    draft_id: str
    request_id: str
    client: str
    status: str = "queued"
    text: str = ""
    attempts: int = 0
    provider: str | None = None
    model: str | None = None
    fallback: bool = False
    error: dict | None = None
    usage: dict | None = None
    elapsed: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    events: list[dict] = field(default_factory=list)


_drafts: dict[str, Draft] = {}
_order: list[str] = []
_lock = threading.Lock()
_changed = threading.Condition(_lock)
_cancellations: dict[str, threading.Event] = {}


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def new_draft(request_id: str, client: str) -> Draft:
    """Створити чернетку зі статусом `queued` і зареєструвати її.
    Роботу не запускає — це справа `start`."""
    draft = Draft(draft_id=uuid.uuid4().hex[:8], request_id=request_id, client=client,
                  created_at=_now(), updated_at=_now())
    with _lock:
        _drafts[draft.draft_id] = draft
        _order.insert(0, draft.draft_id)
        if len(_order) > MAX_DRAFTS:
            for old_id in reversed(_order.copy()):
                if len(_order) <= MAX_DRAFTS:
                    break
                old = _drafts[old_id]
                if old.status in TERMINAL:
                    _order.remove(old_id)
                    _drafts.pop(old_id, None)
                    _cancellations.pop(old_id, None)
    return draft


def get(draft_id: str) -> Draft | None:
    with _lock:
        return _drafts.get(draft_id)


def list_drafts(limit: int = 50) -> list[Draft]:
    """Чернетки, нові першими."""
    with _lock:
        return [_drafts[i] for i in _order[:limit]]


def emit(draft: Draft, event: str, data: dict | None = None) -> None:
    """Додати подію до журналу чернетки й розбудити підписників.

    Статус і текст чернетки ця функція не змінює — це робить фонова
    робота разом із подією, під тим самим замком, якщо їй потрібна
    узгодженість між знімком і потоком.
    """
    with _changed:
        _emit_locked(draft, event, data)


def _emit_locked(draft: Draft, event: str, data: dict | None = None) -> None:
    draft.events.append({"id": len(draft.events) + 1, "event": event, "data": data or {}})
    draft.updated_at = _now()
    _changed.notify_all()


def subscribe(draft_id: str, since: int = 0, heartbeat: float = 15.0) -> Iterator[dict | None]:
    """Віддавати події чернетки, починаючи з номера `since + 1`.

    Спершу — ті, що вже є, потім — нові в міру надходження. Завершується,
    коли чернетка в кінцевому статусі й усі події віддано, або коли
    чернетки немає. Якщо подій немає довше за `heartbeat` секунд, віддає
    `None`: веб-рівень надсилає порожній коментар, щоб зʼєднання не
    закрив проксі і щоб помітити, що клієнт відключився.
    """
    while True:
        with _changed:
            draft = _drafts.get(draft_id)
            if draft is None:
                return
            pending = draft.events[since:]
            if not pending:
                if draft.status in TERMINAL:
                    return
                _changed.wait(timeout=heartbeat)
                pending = draft.events[since:]
        if not pending:
            yield None
            continue
        for event in pending:
            yield event
        since += len(pending)


def start(sample: dict, client: str) -> Draft:
    """Створити чернетку для звернення `sample` і запустити генерацію у
    фоні. Повертає чернетку одразу — найімовірніше, ще `queued`."""
    draft = new_draft(sample["id"], client)
    cancelled = threading.Event()
    with _changed:
        _cancellations[draft.draft_id] = cancelled
        _emit_locked(draft, "status", {"status": "queued", "detail": "чернетка в черзі"})
    worker = threading.Thread(
        target=_run,
        args=(draft, sample, cancelled),
        name=f"draft-{draft.draft_id}",
        daemon=True,
    )
    try:
        worker.start()
    except RuntimeError as exc:
        error = ProviderError("unavailable", "Не вдалося запустити фонову генерацію.", retryable=True)
        _fail(draft, error)
        raise RuntimeError("Не вдалося запустити фонову генерацію.") from exc
    return draft


def cancel(draft_id: str) -> Draft:
    with _changed:
        draft = _drafts.get(draft_id)
        if draft is None:
            raise DraftNotFound(draft_id)
        if draft.status not in {"queued", "running"}:
            raise DraftAlreadyFinished(draft_id)
        event = _cancellations.get(draft_id)
        if event:
            event.set()
        draft.status = "cancelled"
        _emit_locked(draft, "cancelled", {"text": draft.text})
        _cancellations.pop(draft_id, None)
        return draft


class DraftNotFound(Exception):
    pass


class DraftAlreadyFinished(Exception):
    pass


def _set_running(draft: Draft, data: dict) -> bool:
    with _changed:
        if draft.status in TERMINAL:
            return False
        draft.status = "running"
        draft.attempts = max(draft.attempts, int(data.get("attempt", 0)))
        draft.provider = data.get("provider", draft.provider)
        draft.fallback = bool(data.get("fallback", draft.fallback))
        _emit_locked(draft, "status", data)
        return True


def _append_text(draft: Draft, text: str, started: float) -> None:
    if not text:
        return
    with _changed:
        if draft.status in TERMINAL:
            return
        draft.text += text
        if draft.elapsed.get("first_token") is None:
            draft.elapsed["first_token"] = time.monotonic() - started
        _emit_locked(draft, "delta", {"text": text})


def _finish(draft: Draft, result, warnings: list[str], started: float) -> None:
    with _changed:
        if draft.status in TERMINAL:
            return
        draft.status = "done"
        draft.text = result.text
        draft.provider = result.provider
        draft.model = result.model
        draft.usage = result.usage
        draft.elapsed["total"] = time.monotonic() - started
        draft.warnings = warnings
        _emit_locked(draft, "done", {
            "text": draft.text,
            "provider": draft.provider,
            "model": draft.model,
            "usage": draft.usage,
            "elapsed": draft.elapsed,
            "attempts": draft.attempts,
            "fallback": draft.fallback,
            "warnings": draft.warnings,
        })
        _cancellations.pop(draft.draft_id, None)


def _fail(draft: Draft, error: ProviderError, started: float | None = None) -> None:
    with _changed:
        if draft.status in TERMINAL:
            return
        draft.status = "failed"
        draft.error = {
            "kind": error.kind,
            "message": error.message,
            "retryable": error.retryable,
            "request_id": draft.draft_id,
        }
        if started is not None:
            draft.elapsed["total"] = time.monotonic() - started
        _emit_locked(draft, "error", draft.error)
        _cancellations.pop(draft.draft_id, None)


def _run(draft: Draft, sample: dict, cancelled: threading.Event) -> None:
    started = time.monotonic()
    try:
        if not _set_running(draft, {"status": "running", "detail": "генерацію розпочато"}):
            return
        request = drafting.build_request(sample)
        result = gateway.generate(
            request,
            lambda text: _append_text(draft, text, started),
            lambda data: _set_running(draft, data),
            cancelled.is_set,
        )
        warnings = drafting.check(result.text, sample)
        if cancelled.is_set():
            return
        _finish(draft, result, warnings, started)
    except ProviderError as exc:
        if exc.kind == "cancelled" or cancelled.is_set():
            with _changed:
                if draft.status not in TERMINAL:
                    draft.status = "cancelled"
                    _emit_locked(draft, "cancelled", {"text": draft.text})
                    _cancellations.pop(draft.draft_id, None)
            return
        _fail(draft, exc, started)
    except Exception as exc:
        logger.exception("Draft generation failed for %s", draft.draft_id)
        _fail(draft, ProviderError(
            "unknown",
            "Не вдалося створити чернетку через внутрішню помилку.",
            detail=type(exc).__name__,
        ), started)
