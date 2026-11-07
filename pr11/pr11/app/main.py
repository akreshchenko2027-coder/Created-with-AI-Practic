"""HTTP API та сторінка помічника оператора."""

import json
import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import budget, cache, costs, drafting, gateway, jobs, limits, samples, telemetry

telemetry.setup()
log = logging.getLogger("app.http")

app = FastAPI(title="Помічник оператора — ПР11")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"

# Ідентифікатор, який прислав клієнт, приймаємо лише в такому вигляді:
# у журнал не має потрапити довільний рядок із заголовка.
_REQUEST_ID = re.compile(r"^[0-9a-fA-F]{12}$")


class DraftIn(BaseModel):
    """Замовлення чернетки зі сторінки.

    `request_id` — яке звернення зі зразків. `client` — ідентифікатор
    клієнта сторінки для обмеження частоти. `fresh` — оминути кеш:
    оператор хоче іншу чернетку, а не ту саму.
    """

    request_id: str
    client: str = "anonymous"
    fresh: bool = False


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Ідентифікатор запиту в контексті й у відповіді; запис про запит у
    журнал. Для потоку подій тривалість — до початку відповіді, не до
    кінця потоку."""
    incoming = request.headers.get("x-request-id", "")
    request_id = incoming if _REQUEST_ID.match(incoming) else telemetry.new_request_id()
    request.state.request_id = request_id
    token = telemetry.request_id_var.set(request_id)
    started = time.perf_counter()
    status = 500
    try:
        try:
            response = await call_next(request)
        except Exception as exc:
            log.error("http.unhandled", extra={
                "method": request.method,
                "path": request.url.path,
                "error_type": type(exc).__name__,
            })
            response = JSONResponse({
                "error": {
                    "kind": "unknown",
                    "message": "Внутрішня помилка. Передайте ідентифікатор запиту.",
                    "request_id": request_id,
                }
            }, status_code=500)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        log.info("http.request", extra={
            "method": request.method,
            "path": request.url.path,
            "status": status,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "draft_id": draft_match.group(1) if (draft_match := re.search(
                r"/api/drafts/([A-Za-z0-9_-]+)", request.url.path
            )) else None,
        })
        telemetry.request_id_var.reset(token)


def _error_response(kind: str, message: str, status: int, headers: dict | None = None) -> JSONResponse:
    request_id = telemetry.request_id_var.get() or telemetry.new_request_id()
    return JSONResponse(
        {"error": {"kind": kind, "message": message, "request_id": request_id}},
        status_code=status,
        headers=headers,
    )


@app.exception_handler(RequestValidationError)
async def invalid_request(_: Request, __: RequestValidationError) -> JSONResponse:
    return _error_response("bad_request", "Некоректні дані запиту.", 422)


@app.exception_handler(StarletteHTTPException)
async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    messages = {404: "Ресурс не знайдено.", 405: "Метод запиту не підтримується."}
    message = messages.get(exc.status_code, "Запит не вдалося виконати.")
    return _error_response("bad_request", message, exc.status_code, exc.headers)


def draft_to_dict(draft: jobs.Draft) -> dict:
    """Знімок чернетки для сторінки — без журналу подій."""
    data = asdict(draft)
    data["events"] = len(draft.events)
    return data


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """Віддати сторінку."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/health")
def api_health() -> dict:
    """Чинні налаштування — щоб сторінка показала, хто відповідає, яка
    версія prompt і які обмеження діють. Ключа тут немає."""
    return {
        "gateway": gateway.settings(),
        "limits": limits.settings(),
        "cache": cache.settings(),
        "budget": budget.settings(),
        "prompt_version": getattr(drafting, "PROMPT_VERSION", None),
        "log_file": telemetry.settings()["log_file"],
    }


@app.get("/api/usage")
def api_usage() -> dict:
    """Стан витрат і кешу — для панелі на сторінці."""
    return {
        "budget": budget.status(),
        "cache": cache.stats(),
        "pricing": costs.settings(),
    }


@app.get("/api/requests")
def api_requests() -> list[dict]:
    """Зразки звернень для вибору на сторінці."""
    return [samples.public(s) for s in samples.all_samples()]


@app.post("/api/drafts", status_code=202, response_model=None)
def api_create(payload: DraftIn) -> dict | JSONResponse:
    """Замовити чернетку й одразу повернути її стан."""
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,120}", payload.client):
        return _error_response("bad_request", "Некоректний ідентифікатор клієнта.", 422)
    sample = samples.get(payload.request_id)
    if sample is None:
        return _error_response("bad_request", "Звернення не знайдено.", 404)
    try:
        limits.acquire(payload.client)
    except limits.RateLimited as exc:
        return _error_response(
            "rate_limit", str(exc), 429,
            {"Retry-After": str(max(1, int(exc.retry_after + 0.999)))},
        )
    try:
        draft = jobs.start(
            sample, payload.client, fresh=payload.fresh,
            http_request_id=telemetry.request_id_var.get(),
        )
    except RuntimeError:
        return _error_response("unavailable", "Не вдалося запустити генерацію.", 503)
    return draft_to_dict(draft)


@app.get("/api/drafts")
def api_drafts() -> list[dict]:
    """Останні чернетки, нові першими."""
    return [draft_to_dict(d) for d in jobs.list_drafts()]


@app.get("/api/drafts/{draft_id}", response_model=None)
def api_draft(draft_id: str) -> dict | JSONResponse:
    draft = jobs.get(draft_id)
    if draft is None:
        return _error_response("bad_request", "Чернетку не знайдено.", 404)
    return draft_to_dict(draft)


@app.get("/api/drafts/{draft_id}/events", response_model=None)
def api_events(draft_id: str, request: Request) -> StreamingResponse | JSONResponse:
    """Потік подій чернетки у форматі server-sent events.

    Браузер при обриві зʼєднання перепідключається сам і надсилає
    заголовок `Last-Event-ID` — звідси продовжуємо, не повторюючи
    віддане. Порожній коментар раз на кілька секунд тримає зʼєднання
    живим; якщо клієнт відключився, спроба записати в нього завершує
    генератор.
    """
    try:
        since = int(request.headers.get("last-event-id") or 0)
    except ValueError:
        since = 0
    if jobs.get(draft_id) is None:
        return _error_response("bad_request", "Чернетку не знайдено.", 404)

    def stream() -> Iterator[str]:
        for event in jobs.subscribe(draft_id, since):
            if event is None:
                yield ": ping\n\n"
                continue
            data = json.dumps(event["data"], ensure_ascii=False)
            yield f"id: {event['id']}\nevent: {event['event']}\ndata: {data}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/drafts/{draft_id}/cancel", response_model=None)
def api_cancel(draft_id: str) -> dict | JSONResponse:
    try:
        return draft_to_dict(jobs.cancel(draft_id))
    except jobs.DraftNotFound:
        return _error_response("bad_request", "Чернетку не знайдено.", 404)
    except jobs.DraftAlreadyFinished:
        return _error_response("bad_request", "Генерацію вже завершено.", 409)
