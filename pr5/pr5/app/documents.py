import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class Chunk:
    id: str
    text: str
    raw_text: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

def parse_metadata(text: str) -> tuple[dict, str]:
    metadata = {}
    body = text
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if match:
        meta_text = match.group(1)
        body = text[match.end():]
        for line in meta_text.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                metadata[key.strip()] = val.strip()
    return metadata, body

def load_documents(docs_dir: str = "docs") -> list[tuple[str, str, dict]]:
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        docs_path = Path("pr5/docs")
        
    documents = []
    for file_path in docs_path.glob("*.md"):
        if file_path.name.lower() == "readme.md":
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        metadata, body = parse_metadata(content)
        documents.append((file_path.name, body, metadata))
    return documents

def split(text: str, source: str, metadata: dict) -> List[Chunk]:
    chunk_size = int(os.getenv("CHUNK_SIZE", "600"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))
    
    doc_title = metadata.get("title", source)
    context_prefix = f"[окумент: {doc_title}]\n"
    
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[Chunk] = []
    current_chunk = ""
    chunk_idx = 0
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
            full_text = context_prefix + current_chunk.strip()
            chunks.append(Chunk(
                id=f"{source}#{chunk_idx}",
                text=full_text,
                raw_text=current_chunk.strip(),
                source=source,
                metadata=metadata
            ))
            chunk_idx += 1
            if chunk_overlap < len(current_chunk):
                current_chunk = current_chunk[-chunk_overlap:] + "\n\n" + paragraph
            else:
                current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
                
    if current_chunk.strip():
        full_text = context_prefix + current_chunk.strip()
        chunks.append(Chunk(
            id=f"{source}#{chunk_idx}",
            text=full_text,
            raw_text=current_chunk.strip(),
            source=source,
            metadata=metadata
        ))

    return chunks

def load_chunks(docs_dir: str = "docs") -> List[Chunk]:
    docs = load_documents(docs_dir)
    all_chunks = []
    for source, body, metadata in docs:
        all_chunks.extend(split(body, source, metadata))
    return all_chunks
