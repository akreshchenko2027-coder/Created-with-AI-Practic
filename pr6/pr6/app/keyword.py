"""Пошук за ключовими словами — точка порівняння для семантичного.

Той самий набір фрагментів, той самий формат влучень (`Hit`), ті самі
фільтри — інший спосіб ранжувати. Без цього модуля не буде з чим
порівнювати семантичний пошук, а без порівняння — не буде відповіді на
питання, чи він узагалі потрібен для цієї колекції.

Текст токенізується без урахування регістру, зі збереженням номерів і
артикулів. Для ранжування використовується BM25, якщо пакет встановлено;
інакше доступний легкий локальний fallback.
"""

from dataclasses import dataclass, field
from collections import Counter
import math
import re

from .documents import Chunk
from .index import DEFAULT_TOP_K, Hit, matches_filters

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


@dataclass
class KeywordIndex:
    """Індекс для пошуку за словами. Що саме тут зберігати — вирішуєте ви."""

    chunks: list[Chunk]
    extra: dict = field(default_factory=dict)


def tokenize(text: str) -> list[str]:
    """Розбити текст на слова для індексування й для запиту.

    Одна й та сама функція для обох: запит і фрагмент мають розбиватися
    однаково, інакше збігів не буде.
    """
    return re.findall(r"[^\W_]+(?:[-_][^\W_]+)*", text.casefold(), flags=re.UNICODE)


def build(chunks: list[Chunk]) -> KeywordIndex:
    """Зібрати індекс за словами з тих самих фрагментів, що й векторний."""
    tokenized = [tokenize(chunk.text) for chunk in chunks]
    return KeywordIndex(
        chunks=chunks,
        extra={
            "tokenized": tokenized,
            "bm25": BM25Okapi(tokenized) if tokenized and BM25Okapi else None,
        },
    )


def search(
    index: KeywordIndex,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filters: dict | None = None,
) -> list[Hit]:
    """Знайти фрагменти за словами запиту.

    Формат результату — той самий `Hit`, що й у векторного пошуку, з тими
    самими фільтрами за метаданими, щоб сторінка показувала обидва
    способи поруч.
    """
    if top_k <= 0 or not index.chunks:
        return []
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    bm25 = index.extra.get("bm25")
    if bm25 is not None:
        scores = bm25.get_scores(query_tokens)
        tokenized = index.extra.get("tokenized", [])
    else:
        corpus = index.extra.get("tokenized", [])
        tokenized = corpus
        frequencies = Counter(
            token for document in corpus for token in set(document)
        )
        document_count = len(corpus)
        scores = []
        for document in corpus:
            terms = Counter(document)
            score = 0.0
            for token in query_tokens:
                frequency = terms[token]
                if frequency:
                    inverse_frequency = math.log(
                        1 + (document_count - frequencies[token] + 0.5)
                        / (frequencies[token] + 0.5)
                    )
                    score += inverse_frequency * frequency / (frequency + 1)
            scores.append(score)
    ranked = sorted(
        (
            (position, float(score))
            for position, score in enumerate(scores)
            if set(query_tokens).intersection(tokenized[position])
            and matches_filters(index.chunks[position].metadata, filters)
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        Hit(chunk=index.chunks[position], score=score)
        for position, score in ranked[:top_k]
    ]
