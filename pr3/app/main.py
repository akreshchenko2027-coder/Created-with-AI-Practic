"""Веб-рівень застосунку: сторінка зі зверненням і JSON-ендпоінт.

Цей файл не має знати ані про провайдера моделі, ані про те, як
складається запит до неї, — усе це лишається в `app/llm.py`. Тут
вирішується інше: що застосунок віддає клієнтові та з яким HTTP-статусом.

Запуск із папки pr3:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import llm

app = FastAPI(title="Помічник служби підтримки — ПР3")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"
CONTEXT_FILE = Path(__file__).parent.parent / "context.md"


class Question(BaseModel):
    """Звернення користувача."""

    question: str


def load_context() -> str:
    """Прочитати правила організації, на підставі яких відповідає модель.

    Контекст — це дані застосунку, а не знання моделі. Він живе окремим
    файлом і передається в запит разом зі зверненням.
    """
    return CONTEXT_FILE.read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Віддати сторінку зі зверненням."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.post("/api/ask")
def api_ask(payload: Question):
    """Повернути відповідь помічника у форматі JSON.

    Зараз виняток із модуля роботи з моделлю не обробляється — застосунок
    просто впаде з помилкою 500. Спроєктуйте обробку самі: які збої
    можливі (таймаут, перевищення ліміту, недоступність сервісу, невірний
    ключ, порожнє звернення), який HTTP-статус відповідає кожному з них
    і що в такому разі побачить користувач.

    Зверніть увагу: технічний текст помилки провайдера — не те, що варто
    показувати користувачеві, але те, що варто записати в журнал.
    """
    return llm.ask(payload.question, load_context())
