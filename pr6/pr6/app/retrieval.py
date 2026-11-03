"""Відбір фрагментів для моделі та збирання контексту.

Між пошуком із ПР5 і мовною моделлю стоїть шар, якого в ПР5 не було: він
вирішує, *що саме* модель побачить. Пошук повертає влучення з оцінками —
стільки, скільки попросили; модель потребує тексту — обмеженого за
розміром, упорядкованого, з позначками джерел, на які вона зможе
послатися. Цей шар — тут.

Векторний і лексичний пошук зливаються за reciprocal rank fusion.
Внутрішні та архівні документи відсікаються незмінними фільтрами, а
контекст маркується перевірними номерами джерел.
"""

import os
import math
from dataclasses import dataclass

from dotenv import load_dotenv

from .documents import Chunk
from .embeddings import embed_query
from .index import DEFAULT_TOP_K, Hit, SearchIndex, search as semantic_search
from .keyword import KeywordIndex
from .keyword import search as keyword_search

load_dotenv()

# Скільки фрагментів іде в контекст моделі й скільки токенів на них є.
# Значення — відправна точка; їхнє обґрунтування — результат вашої
# перевірки на наборі запитань, а не здогадка.
CONTEXT_CHUNKS = int(os.getenv("RAG_CONTEXT_CHUNKS", "4"))
CONTEXT_BUDGET = int(os.getenv("RAG_CONTEXT_BUDGET", "1500"))
if CONTEXT_CHUNKS < 1:
    raise ValueError("RAG_CONTEXT_CHUNKS має бути щонайменше 1")
if CONTEXT_BUDGET < 1:
    raise ValueError("RAG_CONTEXT_BUDGET має бути щонайменше 1")


@dataclass
class Source:
    """Джерело, показане моделі й користувачеві.

    `ref` — номер, під яким фрагмент стоїть у контексті й на який модель
    посилається у відповіді. `chunk` — сам фрагмент із метаданими;
    `score` — оцінка пошуку, за якою його відібрано.
    """

    ref: int
    chunk: Chunk
    score: float


def retrieve(
    query: str,
    index: SearchIndex,
    keyword_index: KeywordIndex | None,
    filters: dict | None = None,
) -> list[Hit]:
    """Поєднати векторний і BM25-пошук та застосувати незмінні фільтри доступу."""
    allowed_filters = {"product", "category"}
    requested = filters or {}
    unknown = set(requested) - allowed_filters
    if unknown:
        raise ValueError(f"Непідтримуваний фільтр: {', '.join(sorted(unknown))}")
    safe_filters = {"audience": "клієнти", "status": "чинний", **requested}
    search_k = max(DEFAULT_TOP_K, CONTEXT_CHUNKS * 4)
    vector_hits = semantic_search(query_vector=embed_query(query), index=index,
                                  top_k=search_k, filters=safe_filters)
    lexical_hits = (
        keyword_search(keyword_index, query, top_k=search_k, filters=safe_filters)
        if keyword_index is not None
        else []
    )

    merged: dict[tuple[str, str, str], tuple[Chunk, float]] = {}
    for hits in (vector_hits, lexical_hits):
        for rank, hit in enumerate(hits, start=1):
            key = (hit.chunk.source, str(hit.chunk.metadata.get("heading", "")), hit.chunk.text)
            chunk, score = merged.get(key, (hit.chunk, 0.0))
            merged[key] = (chunk, score + 1 / (60 + rank))

    ranked = sorted(merged.values(), key=lambda item: item[1], reverse=True)
    selected: list[Hit] = []
    per_document: dict[str, int] = {}
    for pass_number in (0, 1):
        for chunk, score in ranked:
            if per_document.get(chunk.source, 0) != pass_number:
                continue
            selected.append(Hit(chunk=chunk, score=score))
            per_document[chunk.source] = pass_number + 1
            if len(selected) == CONTEXT_CHUNKS:
                return selected
    return selected


def _estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 3))


def build_context(hits: list[Hit], budget: int = CONTEXT_BUDGET) -> tuple[str, list[Source]]:
    """Зібрати з влучень текст контексту для моделі та перелік джерел.

    Кожен фрагмент позначається стабільним номером; надто великі фрагменти
    пропускаються, щоб пізніші корисні фрагменти також могли вміститися.
    """
    if budget <= 0:
        raise ValueError("Бюджет контексту має бути додатним")
    blocks: list[str] = []
    sources: list[Source] = []
    used_tokens = 0
    for hit in hits:
        metadata = hit.chunk.metadata
        details = [
            f"файл: {hit.chunk.source}",
            f"назва: {metadata.get('title', hit.chunk.source)}",
        ]
        if metadata.get("heading"):
            details.append(f"розділ: {metadata['heading']}")
        if metadata.get("updated"):
            details.append(f"редакція: {metadata['updated']}")
        ref = len(sources) + 1
        block = f"[{ref}] {' | '.join(details)}\n{hit.chunk.text}"
        cost = _estimate_tokens(block)
        if used_tokens + cost > budget:
            continue
        blocks.append(block)
        sources.append(Source(ref=ref, chunk=hit.chunk, score=hit.score))
        used_tokens += cost
    return "\n\n".join(blocks), sources
