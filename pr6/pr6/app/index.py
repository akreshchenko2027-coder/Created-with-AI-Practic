"""Векторний індекс: зберігання векторів фрагментів і пошук найближчих.

Індекс — це вектори всіх фрагментів, самі фрагменти з метаданими й назва
моделі, якою вектори отримано. Він будується окремою командою
(`ingest.py`) і зберігається на диску, а застосунок при старті лише
читає його: перераховувати ембедінги колекції на кожен запуск — марна
трата часу, а на кожен запит — тим паче.

Пошук використовує косинусну схожість нормалізованих векторів; фрагменти
й модель зберігаються окремо у `index.json` та `vectors.npy`.
"""

import os
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from .documents import Chunk

load_dotenv()

INDEX_DIR = Path(__file__).parent.parent / "index"

# Скільки результатів повертати за замовчуванням і чи відсікати слабкі.
# Порожній поріг означає «без порога».
DEFAULT_TOP_K = int(os.getenv("SEARCH_TOP_K", "5"))
_threshold = os.getenv("SIMILARITY_THRESHOLD", "0.35").strip()
SIMILARITY_THRESHOLD: float | None = float(_threshold) if _threshold else None
if SIMILARITY_THRESHOLD is not None and not -1 <= SIMILARITY_THRESHOLD <= 1:
    raise ValueError("SIMILARITY_THRESHOLD має бути від -1 до 1")


class FilterError(ValueError):
    """Фільтр не відповідає полям метаданих документів."""


@dataclass
class Hit:
    """Одне влучення пошуку: фрагмент і оцінка його схожості із запитом."""

    chunk: Chunk
    score: float


@dataclass
class SearchIndex:
    """Індекс у памʼяті.

    `vectors` — масив (кількість фрагментів × розмірність); рядок i
    відповідає `chunks[i]`. `model_name` — модель, якою отримано вектори:
    без неї індекс, збудований однією моделлю, мовчки шукатиме векторами
    іншої.
    """

    chunks: list[Chunk]
    vectors: np.ndarray
    model_name: str
    extra: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.chunks)


def build(chunks: list[Chunk], vectors: np.ndarray, model_name: str) -> SearchIndex:
    """Зібрати індекс із фрагментів і їхніх векторів.

    Тут перевіряється, що векторів стільки ж, скільки фрагментів, і
    вирішується, що зробити з векторами перед пошуком (наприклад,
    нормалізувати, якщо схожість — косинусна).
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("Вектори мають бути двовимірним масивом")
    if vectors.shape[0] != len(chunks):
        raise ValueError("Кількість векторів не збігається з кількістю фрагментів")
    if not np.isfinite(vectors).all():
        raise ValueError("Вектори містять NaN або нескінченні значення")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
    return SearchIndex(chunks=chunks, vectors=normalized, model_name=model_name)


def save(index: SearchIndex, path: Path = INDEX_DIR) -> None:
    """Зберегти індекс на диск.

    Формат — ваше рішення: `numpy` для векторів і JSON для фрагментів з
    метаданими — цілком достатньо. Назву моделі зберігайте обовʼязково.
    """
    path.mkdir(parents=True, exist_ok=True)
    vectors_path = path / "vectors.npy"
    metadata_path = path / "index.json"
    vectors_tmp = path / "vectors.npy.tmp"
    metadata_tmp = path / "index.json.tmp"
    with vectors_tmp.open("wb") as file:
        np.save(file, index.vectors, allow_pickle=False)
    metadata_tmp.write_text(
        json.dumps(
            {
                "model_name": index.model_name,
                "chunks": [
                    {"text": chunk.text, "source": chunk.source, "metadata": chunk.metadata}
                    for chunk in index.chunks
                ],
                "extra": index.extra,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(vectors_tmp, vectors_path)
    os.replace(metadata_tmp, metadata_path)


def load(path: Path = INDEX_DIR) -> SearchIndex:
    """Прочитати індекс із диска.

    Якщо індексу немає — підняти зрозумілу помилку: веб-рівень має
    сказати користувачеві «індекс не збудовано», а не впасти з
    `FileNotFoundError` десь усередині.
    """
    vectors_path = path / "vectors.npy"
    metadata_path = path / "index.json"
    if not vectors_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Індекс не знайдено у {path}; виконайте python ingest.py")
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        vectors = np.load(vectors_path, allow_pickle=False)
        chunks = [
            Chunk(
                text=item["text"],
                source=item["source"],
                metadata=item.get("metadata", {}),
            )
            for item in payload["chunks"]
        ]
        model_name = payload["model_name"]
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Не вдалося прочитати індекс у {path}: {exc}") from exc
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise ValueError(f"Пошкоджений індекс у {path}: кількість векторів і фрагментів різна")
    return SearchIndex(
        chunks=chunks,
        vectors=np.asarray(vectors, dtype=np.float32),
        model_name=model_name,
        extra=payload.get("extra", {}),
    )


def matches_filters(metadata: dict, filters: dict | None) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        if key not in metadata:
            raise FilterError(f"Невідомий фільтр «{key}»")
        if str(metadata[key]).strip().casefold() != str(expected).strip().casefold():
            return False
    return True


def search(
    index: SearchIndex,
    query_vector: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
    filters: dict | None = None,
    threshold: float | None = SIMILARITY_THRESHOLD,
) -> list[Hit]:
    """Знайти фрагменти, найближчі до вектора запиту.

    `filters` — умови на метадані фрагмента, наприклад
    `{"category": "інструкція", "audience": "клієнти"}`. Як трактувати
    поле з датою і що робити з невідомим полем — ваше рішення.

    Повертає щонайбільше `top_k` влучень, упорядкованих за спаданням
    оцінки. Оцінка — та сама міра схожості, що й для ранжування, і саме
    вона показується користувачеві.
    """
    if top_k <= 0:
        return []
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    if query.size != index.vectors.shape[1]:
        raise ValueError(
            f"Розмірність запиту ({query.size}) не збігається з індексом "
            f"({index.vectors.shape[1]})"
        )
    norm = np.linalg.norm(query)
    if norm == 0:
        return []
    query = query / norm

    candidates = [
        position for position, chunk in enumerate(index.chunks)
        if matches_filters(chunk.metadata, filters)
    ]
    if not candidates:
        return []
    scores = index.vectors[candidates] @ query
    ranked = sorted(zip(candidates, scores, strict=True), key=lambda item: float(item[1]), reverse=True)
    hits: list[Hit] = []
    for position, raw_score in ranked:
        score = float(raw_score)
        if threshold is not None and score < threshold:
            continue
        hits.append(Hit(chunk=index.chunks[position], score=score))
        if len(hits) == top_k:
            break
    return hits
