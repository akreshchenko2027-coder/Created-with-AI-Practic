import sys
from pathlib import Path

# Додаємо теку pr5 до шляхів пошуку модулів Python
sys.path.insert(0, str(Path(__file__).parent))

from app import documents, embeddings, index

def main():
    print("Завантаження та обробка документів...")
    docs_dir = "pr5/docs" if Path("pr5/docs").exists() else "docs"

    chunks = documents.load_chunks(docs_dir)
    print(f"Знайдено фрагментів: {len(chunks)}")

    print(f"Генерація векторних ембеддінгів (модель: {embeddings.MODEL_NAME})...")
    vectors = embeddings.embed_passages([chunk.text for chunk in chunks])

    print("Побудова та збереження індексу...")
    built_index = index.build(chunks, vectors, embeddings.MODEL_NAME)
    index.save(built_index)
    print("Індексацію успішно завершено!")

if __name__ == "__main__":
    main()
