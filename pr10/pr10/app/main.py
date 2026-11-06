import json
import logging
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import gateway, jobs, limits, samples

app = FastAPI(title="Помічник оператора — ПР10")
logger = logging.getLogger(__name__)

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


class DraftIn(BaseModel):
    request_id: str = Field(min_length=1, max_length=80)
    client: str = Field(default="anonymous", min_length=1, max_length=120)


def error_response(
    kind: str,
    message: str,
    status_code: int,
    *,
    retryable: bool = False,
    request_id: str | None = None,
    retry_after: float | None = None,
) -> JSONResponse:
    headers = {"Retry-After": str(max(1, int(retry_after + 0.999)))} if retry_after is not None else None
    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content={
            "error": {
                "kind": kind,
                "message": message,
                "retryable": retryable,
                "request_id": request_id or uuid.uuid4().hex[:12],
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, __: RequestValidationError) -> JSONResponse:
    return error_response("bad_request", "Перевірте поля запиту й спробуйте ще раз.", 422)


def draft_to_dict(draft: jobs.Draft) -> dict:
    data = asdict(draft)
    data["events"] = len(draft.events)
    return data


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/health")
def api_health() -> dict:
    return {"gateway": gateway.settings(), "limits": limits.settings()}


@app.get("/api/requests")
def api_requests() -> list[dict]:
    return [samples.public(s) for s in samples.all_samples()]


@app.post("/api/drafts", status_code=202, response_model=None)
def api_create(payload: DraftIn) -> dict | JSONResponse:
    sample = samples.get(payload.request_id)
    if sample is None:
        return error_response("not_found", "Звернення не знайдено.", 404)
    client = payload.client.strip()
    if not client:
        return error_response("bad_request", "Потрібно вказати клієнта.", 422)
    try:
        limits.acquire(client)
    except limits.RateLimited as exc:
        return error_response(
            "rate_limit",
            "Ліміт замовлень вичерпано. Спробуйте пізніше.",
            429,
            retryable=True,
            retry_after=exc.retry_after,
        )
    try:
        return draft_to_dict(jobs.start(sample, client))
    except Exception:
        logger.exception("Could not start draft for request %s", payload.request_id)
        return error_response("unavailable", "Не вдалося поставити чернетку в чергу.", 503, retryable=True)


@app.get("/api/drafts")
def api_drafts() -> list[dict]:
    return [draft_to_dict(d) for d in jobs.list_drafts()]


@app.get("/api/drafts/{draft_id}", response_model=None)
def api_draft(draft_id: str) -> dict | JSONResponse:
    draft = jobs.get(draft_id)
    if draft is None:
        return error_response("not_found", "Чернетку не знайдено.", 404)
    return draft_to_dict(draft)


@app.get("/api/drafts/{draft_id}/events", response_model=None)
def api_events(draft_id: str, request: Request) -> StreamingResponse | JSONResponse:
    if jobs.get(draft_id) is None:
        return error_response("not_found", "Чернетку не знайдено.", 404)
    try:
        since = int(request.headers.get("last-event-id") or 0)
    except ValueError:
        since = 0
    since = max(0, since)

    def stream() -> Iterator[str]:
        for event in jobs.subscribe(draft_id, since):
            if event is None:
                yield ": ping\n\n"
                continue
            data = json.dumps(event["data"], ensure_ascii=False)
            yield f"id: {event['id']}\nevent: {event['event']}\ndata: {data}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/drafts/{draft_id}/cancel")
def api_cancel(draft_id: str) -> dict:
    try:
        return draft_to_dict(jobs.cancel(draft_id))
    except jobs.DraftNotFound:
        return error_response("not_found", "Чернетку не знайдено.", 404)
    except jobs.DraftAlreadyFinished:
        return error_response("conflict", "Чернетка вже завершена; скасувати її неможливо.", 409)
