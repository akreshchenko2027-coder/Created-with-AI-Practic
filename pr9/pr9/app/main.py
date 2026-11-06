"""Веб-рівень застосунку: сторінка звернень, черга оператора і
JSON-ендпоінти.

Цей файл не знає ані які кроки є у workflow, ані як зберігається
стан, ані якою моделлю розібрано звернення — усе це в
`app/workflow.py`, `app/store.py` і `app/llm.py`. Тут вирішується
інше: що застосунок приймає від сторінки, що віддає їй і з яким
HTTP-статусом.

Для порівняння той самий текст звернення можна віддати агентові з ПР8
(`app/assistant.py`) — сторінка має перемикач способу обробки.

Запуск із папки pr9:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from shop import service

from . import assistant, llm, store, workflow


@asynccontextmanager
async def lifespan(_: FastAPI):
    workflow.recover_incomplete()
    yield


app = FastAPI(title="Звернення клієнтів — ПР9", lifespan=lifespan)

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


class RequestIn(BaseModel):
    """Нове звернення зі сторінки.

    `customer_id` — клієнт, обраний на сторінці. Тут він заміняє вхід у
    кабінет: у справжньому застосунку веб-рівень узяв би його із сесії.
    `mode` — хто обробляє: `workflow` або `agent` (помічник із ПР8).
    """

    customer_id: str = Field(min_length=1, max_length=40)
    text: str = Field(min_length=1, max_length=2000)
    mode: Literal["workflow", "agent"] = "workflow"


class DecisionIn(BaseModel):
    """Рішення оператора щодо дії, яка чекає підтвердження."""

    approve: bool
    comment: str = Field(default="", max_length=500)


def run_to_dict(run: workflow.Run) -> dict:
    """Перетворити стан звернення на те, що піде на сторінку."""
    data = asdict(run)
    data["mode"] = "workflow"
    data["limits"] = workflow.limits()
    return data


def answer_to_dict(result: assistant.Answer) -> dict:
    """Перетворити результат агента з ПР8 на те, що піде на сторінку.

    Відповідь зберігає сумісність зі сторінкою помічника ПР8.
    """
    return {
        "mode": "agent",
        "answer": result.text,
        "calls": [asdict(call) for call in result.calls],
        "rounds": result.rounds,
        "stopped": result.stopped,
        "model": result.model,
        "elapsed": result.elapsed,
        "usage": result.usage,
    }


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """Віддати сторінку."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/customers")
def api_customers() -> list[dict]:
    """Клієнти, від імені яких можна «увійти» на сторінці."""
    return service.list_customers()


@app.post("/api/reset")
def api_reset() -> dict:
    """Повернути дані магазину до початкового стану: створені повернення
    й скасування зникають. Збережені звернення не зачіпаються — їх
    видаляйте з `RUNS_DIR` самі, якщо потрібно."""
    service.reset()
    return {"ok": True}


@app.post("/api/requests")
def api_create(payload: RequestIn) -> dict:
    """Прийняти звернення й обробити його обраним способом.

    Workflow повертає стан звернення (див. `run_to_dict`), агент —
    відповідь і журнал викликів, як у ПР8 (див. `answer_to_dict`).

    Порожній текст і невідомий клієнт відхиляються до виклику моделі.
    Помилки моделі, сховища й вводу повертаються з відповідними
    HTTP-статусами, а контрольовані завершення workflow — як стан Run.
    """
    customer_id = payload.customer_id.strip()
    text = payload.text.strip()
    if not customer_id:
        raise HTTPException(status_code=422, detail="Потрібно вибрати клієнта.")
    if not text:
        raise HTTPException(status_code=422, detail="Введіть текст звернення.")
    if customer_id not in {item["customer_id"] for item in service.list_customers()}:
        raise HTTPException(status_code=404, detail="Клієнта не знайдено.")
    if payload.mode == "agent":
        try:
            return answer_to_dict(assistant.answer(text, customer_id))
        except llm.LLMError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    try:
        return run_to_dict(workflow.start(text, customer_id))
    except store.StoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/runs")
def api_runs(status: str | None = None) -> list[dict]:
    """Звернення, оброблені workflow, — усі або з певним статусом.

    Черга оператора на сторінці — це `?status=awaiting_approval`. Якщо
    ваш статус очікування називається інакше, змініть його в сторінці.
    """
    try:
        return [run_to_dict(run) for run in store.list_runs(status)]
    except store.StoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}")
def api_run(run_id: str) -> dict:
    try:
        run = store.load(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except store.StoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if run is None:
        raise HTTPException(status_code=404, detail="Звернення не знайдено.")
    return run_to_dict(run)


@app.post("/api/runs/{run_id}/decision")
def api_decision(run_id: str, payload: DecisionIn) -> dict:
    """Рішення оператора: підтвердити чи відхилити дію.

    Повторне рішення не запускає дію вдруге; невідоме звернення й
    помилки сховища повертаються з відповідними HTTP-статусами.
    """
    try:
        return run_to_dict(workflow.resume(run_id, payload.approve, payload.comment))
    except workflow.RunNotFound as exc:
        raise HTTPException(status_code=404, detail="Звернення не знайдено.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except store.StoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
