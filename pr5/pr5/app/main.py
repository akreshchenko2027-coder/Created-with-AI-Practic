"""Веб-рівень застосунку: сторінка пошуку і JSON-ендпоінти.

Цей файл не знає ані якою моделлю отримано вектори, ані як влаштований
індекс, ані як ранжує пошук за словами — усе це лишається в модулях
`app/embeddings.py`, `app/index.py`, `app/keyword.py`. Тут вирішується
інше: що застосунок приймає від сторінки, що віддає їй і з яким
HTTP-статусом.

Індекс будується заздалегідь командою `python ingest.py` (з папки pr5),
а тут лише читається при старті.

Запуск із папки pr5:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import embeddings, index, bm25_keyword as keyword

app = FastAPI(title="Пошук у базі знань — ПР5")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


class SearchRequest(BaseModel):
    """Те, що надсилає сторінка.

    `mode` — `semantic`, `keyword` або `both`. `filters` — умови на
    метадані фрагментів, наприклад `{"category": "інструкція"}`; порожній
    словник означає «без фільтрів». `threshold` — поріг схожості для
    семантичного пошуку; `None` — узяти значення з конфігурації.
    """

    query: str
    mode: str = "both"
    top_k: int = index.DEFAULT_TOP_K
    filters: dict = {}
    threshold: float | None = None


def hit_to_dict(hit: index.Hit) -> dict:
    """Перетворити влучення на те, що піде на сторінку."""
    return {
        "score": hit.score,
        "text": hit.chunk.text,
        "source": hit.chunk.source,
        "metadata": hit.chunk.metadata,
    }


@app.on_event("startup")
def load_indexes() -> None:
    """Прочитати збудований індекс і зібрати індекс за словами з тих
    самих фрагментів.

    Якщо індексу на диску немає, застосунок усе одно стартує: сторінка
    має відкритися й пояснити, що робити. Як саме повідомити про це
    користувачеві — вирішуєте ви; каркас лише лишає `app.state.index`
    порожнім.
    """
    app.state.index = None
    app.state.keyword_index = None
    try:
        app.state.index = index.load()
    except Exception as exc:  # noqa: BLE001 — старт не має падати без індексу
        print(f"Індекс не завантажено: {type(exc).__name__}: {exc}")
        return
    app.state.keyword_index = keyword.build(app.state.index.chunks)


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """Віддати сторінку пошуку."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/status")
def api_status() -> dict:
    """Стан індексу: чи збудовано, скільки фрагментів, якою моделлю.

    Сторінка викликає це при відкритті, щоб показати підказку, якщо
    індексу ще немає.
    """
    idx = app.state.index
    if idx is None:
        return {"ready": False, "hint": "індекс не збудовано — виконайте python ingest.py"}
    sources = {chunk.source for chunk in idx.chunks}
    return {
        "ready": True,
        "chunks": len(idx),
        "documents": len(sources),
        "model": idx.model_name,
    }


@app.post("/api/search")
def api_search(payload: SearchRequest) -> dict:
    """Виконати пошук і повернути влучення обох способів.

    Сторінка очікує обʼєкт із полями `semantic` і `keyword` (список
    влучень або `null`, якщо спосіб не запитували) та `elapsed` — час
    кожного способу в секундах.

    Збої тут не оброблено. Порожній запит, відсутній індекс, невідомий
    режим, фільтр за полем, якого немає в метаданих, — усе це поки що
    закінчується помилкою 500. Який статус і яке повідомлення має
    отримати сторінка в кожному випадку — вирішуєте ви.
    """
    idx = app.state.index
    result: dict = {"query": payload.query, "semantic": None, "keyword": None, "elapsed": {}}

    # Якщо індекс ще не завантажено в стані застосунку, спробуємо підвантажити його з диска.
    # Це дозволить уникнути перезапуску сервера після побудови індексу через `ingest.py`.
    if idx is None:
        idx = index.load()
        if idx is None:
            raise HTTPException(status_code=503, detail="індекс не збудовано — виконайте python ingest.py")
        app.state.index = idx
        app.state.keyword_index = keyword.build(app.state.index.chunks)

    if payload.mode in ("semantic", "both"):
        started = time.perf_counter()
        vector = embeddings.embed_query(payload.query)
        hits = index.search(
            idx, vector,
            top_k=payload.top_k,
            **(payload.filters or {}),
            threshold=payload.threshold if payload.threshold is not None
            else index.SIMILARITY_THRESHOLD,
        )
        result["elapsed"]["semantic"] = time.perf_counter() - started
        result["semantic"] = [hit_to_dict(h) for h in hits]

    if payload.mode in ("keyword", "both"):
        started = time.perf_counter()
        hits = keyword.search(
            app.state.keyword_index, payload.query,
            top_k=payload.top_k,
            **(payload.filters or {}),
        )
        result["elapsed"]["keyword"] = time.perf_counter() - started
        result["keyword"] = [hit_to_dict(h) for h in hits]

    return result
