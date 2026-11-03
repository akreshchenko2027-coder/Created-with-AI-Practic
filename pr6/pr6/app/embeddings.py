"""Модуль ембедінгів: єдине місце застосунку, яке знає, яка модель
перетворює текст на вектор і як її викликати.

Індекс (`app/index.py`) і веб-рівень (`app/main.py`) отримують звідси
готові вектори й не знають, локальна це модель чи API. Замінивши модель
тут, ви не змінюєте решту коду — але маєте перебудувати індекс: вектори
різних моделей непорівнянні.
Модель вантажиться лише за першого кодування; для multilingual-e5
використовуються префікси `passage:` і `query:`, а вектори нормалізуються.
"""

import os
from functools import lru_cache

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Назва моделі ембедінгів. Для локальної моделі — ідентифікатор на
# Hugging Face Hub; для моделі через API — назва в провайдера.
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")


def get_model():
    """Повернути готову до роботи модель.

    Локальна модель вантажиться з диска секунди й займає сотні мегабайт
    памʼяті — створюйте її один раз, а не на кожен запит. Для моделі через
    API тут створюється клієнт (ключ — із `.env`, як у ПР3).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Не встановлено sentence-transformers; виконайте pip install -r requirements.txt"
        ) from exc
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def _model():
    return get_model()


def embed_passages(texts: list[str]) -> np.ndarray:
    """Перетворити тексти фрагментів на вектори.

    Повертає масив розміру (кількість текстів × розмірність моделі).
    Викликається під час індексування — для всієї колекції одразу.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    vectors = _model().encode(
        [f"passage: {text}" for text in texts],
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 32,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Перетворити запит користувача на вектор тієї самої розмірності.

    Окрема функція навмисно: у моделей із префіксами запит кодується не
    так, як фрагмент, і саме тут це видно.
    """
    vector = _model().encode(
        f"query: {text}",
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vector, dtype=np.float32).reshape(-1)
