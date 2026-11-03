"""Колекція документів: читання, метадані, поділ на фрагменти.

Модуль читає метадані з Markdown і ділить текст за заголовками, абзацами
та реченнями. Решта застосунку працює з готовими фрагментами (`Chunk`) і
не читає файлів напряму.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = Path(__file__).parent.parent / "docs"

# Параметри поділу — відправна точка, а не рекомендація. У чому їх
# рахувати (символи, слова, токени моделі) і як застосовувати, коли
# ділите за заголовками, — ваше рішення.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
if CHUNK_SIZE < 100:
    raise ValueError("CHUNK_SIZE має бути не меншим за 100 символів")
if CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError("CHUNK_OVERLAP має бути від 0 до CHUNK_SIZE - 1")


@dataclass
class Chunk:
    """Фрагмент документа — одиниця індексування й пошуку.

    `text` — те, що перетворюється на вектор і показується в результатах.
    `source` — імʼя файлу, з якого взято фрагмент.
    `metadata` — поля з блоку метаданих файлу (title, category, product,
    audience, updated, status) плюс те, що ви вирішите додати самі:
    заголовок розділу, порядковий номер фрагмента, позицію в документі.
    Фільтри пошуку працюють саме з цим словником.
    """

    text: str
    source: str
    metadata: dict = field(default_factory=dict)


def parse_front_matter(raw: str) -> tuple[dict, str]:
    """Відокремити блок метаданих від тексту документа.

    Блок — рядки `ключ: значення` між двома рядками `---` на початку
    файлу. Повертає словник метаданих і решту тексту. Порожні значення
    (`product:` без нічого) стають порожнім рядком. Якщо блоку немає —
    порожній словник і текст як є.
    """
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw
    metadata: dict = {}
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            body = "\n".join(lines[i + 1:]).lstrip("\n")
            return metadata, body
        if ":" in line:
            key, _, value = line.partition(":")
            metadata[key.strip()] = value.strip()
    return {}, raw


def load_documents(docs_dir: Path = DOCS_DIR) -> list[tuple[str, dict, str]]:
    """Прочитати всі документи колекції.

    Повертає список трійок (імʼя файлу, метадані, текст) для кожного
    `*.md` у папці, крім `README.md` — він описує колекцію, а не є її
    частиною. Порядок — за іменем файлу, щоб індекс будувався однаково
    від запуску до запуску.
    """
    documents = []
    for path in sorted(docs_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        metadata, body = parse_front_matter(path.read_text(encoding="utf-8"))
        documents.append((path.name, metadata, body))
    return documents


def split(text: str, source: str, metadata: dict) -> list[Chunk]:
    """Ділити спочатку за заголовками й абзацами, а довгі абзаци — за реченнями."""
    title = str(metadata.get("title") or source)
    chunks: list[Chunk] = []
    heading = ""
    section_parts: list[str] = []

    def add_piece(piece: str) -> None:
        body = piece.strip()
        if not body:
            return
        prefix = f"{title}\n{heading}\n".strip()
        chunk_text = f"{prefix}\n{body}" if prefix else body
        chunk_metadata = {**metadata, "title": title}
        if heading:
            chunk_metadata["heading"] = heading
        chunks.append(Chunk(text=chunk_text, source=source, metadata=chunk_metadata))

    def flush_section() -> None:
        if not section_parts:
            return
        paragraphs = "\n\n".join(section_parts).strip()
        section_parts.clear()
        if not paragraphs:
            return

        current = ""
        for paragraph in re.split(r"\n\s*\n", paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if len(paragraph) > CHUNK_SIZE:
                if current:
                    add_piece(current)
                    current = ""
                sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                for sentence in sentences:
                    if len(sentence) <= CHUNK_SIZE:
                        if current and len(current) + len(sentence) + 1 <= CHUNK_SIZE:
                            current = f"{current} {sentence}"
                        else:
                            if current:
                                add_piece(current)
                            current = sentence
                        continue

                    if current:
                        add_piece(current)
                        current = ""
                    start = 0
                    while start < len(sentence):
                        end = min(start + CHUNK_SIZE, len(sentence))
                        if end < len(sentence):
                            boundary = sentence.rfind(" ", start, end)
                            if boundary > start:
                                end = boundary
                        add_piece(sentence[start:end])
                        if end == len(sentence):
                            break
                        start = max(start + 1, end - CHUNK_OVERLAP)
                continue

            if current and len(current) + len(paragraph) + 2 > CHUNK_SIZE:
                add_piece(current)
                overlap_size = min(CHUNK_OVERLAP, max(0, CHUNK_SIZE - len(paragraph) - 2))
                overlap = current[-overlap_size:] if overlap_size else ""
                current = f"{overlap}\n\n{paragraph}" if overlap else paragraph
            else:
                current = f"{current}\n\n{paragraph}" if current else paragraph
        if current:
            add_piece(current)

    for line in text.splitlines():
        match = re.match(r"^#{2,3}\s+(.+?)\s*#*\s*$", line)
        if match:
            flush_section()
            heading = match.group(1).strip()
        else:
            section_parts.append(line)
    flush_section()
    return chunks


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    """Прочитати колекцію й повернути всі її фрагменти."""
    chunks: list[Chunk] = []
    for source, metadata, body in load_documents(docs_dir):
        chunks.extend(split(body, source, metadata))
    return chunks
