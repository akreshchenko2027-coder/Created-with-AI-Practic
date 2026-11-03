"""Побудова індексу з колекції документів.

Запуск із папки pr6:
    python ingest.py

Читає `docs/`, ділить документи на фрагменти, рахує вектори й зберігає
індекс на диск. Виконується один раз після кожної зміни колекції, моделі
або правила поділу — застосунок при старті лише читає готовий індекс.

Сам скрипт — плумбінг: він викликає функції модулів у правильному
порядку й друкує, що вийшло. Уся суть — у `app/documents.py` (поділ),
`app/embeddings.py` (вектори) та `app/index.py` (збереження). Доки ці
функції лишаються заготовками, скрипт зупиниться на першій із них.
"""

import sys
import time
from collections import Counter

from app import documents, embeddings, index


def main() -> int:
    started = time.perf_counter()
    docs = documents.load_documents()
    print(f"Документів: {len(docs)}")

    chunks = documents.load_chunks()
    per_doc = Counter(chunk.source for chunk in chunks)
    lengths = [len(chunk.text) for chunk in chunks]
    print(f"Фрагментів: {len(chunks)}")
    if chunks:
        print(f"  довжина, символів: min {min(lengths)}, "
              f"середня {sum(lengths) // len(lengths)}, max {max(lengths)}")
        for source, count in sorted(per_doc.items()):
            print(f"  {count:3d}  {source}")

    print(f"Модель: {embeddings.MODEL_NAME}")
    t_embed = time.perf_counter()
    vectors = embeddings.embed_passages([chunk.text for chunk in chunks])
    print(f"Вектори: {vectors.shape[0]} x {vectors.shape[1]} "
          f"за {time.perf_counter() - t_embed:.1f} с")

    built = index.build(chunks, vectors, embeddings.MODEL_NAME)
    index.save(built)
    print(f"Індекс збережено у {index.INDEX_DIR} "
          f"({time.perf_counter() - started:.1f} с загалом)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
