from typing import List, Optional
from rank_bm25 import BM25Okapi
from .documents import Chunk
from .index import Index, Hit, DEFAULT_TOP_K

class KeywordIndex:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        corpus = [chunk.text.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(corpus)

def build(chunks: List[Chunk]) -> KeywordIndex:
    """Створення індексу BM25 для пошуку за ключовими словами."""
    return KeywordIndex(chunks)

def search(
    index: Index | KeywordIndex,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    category: Optional[str] = None,
    product: Optional[str] = None,
    audience: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Hit]:
    """Пошук за ключовими словами за допомогою BM25."""
    if not query or not query.strip():
        return []

    bm25_obj = getattr(index, "bm25", None)
    chunks_list = getattr(index, "chunks", [])

    if bm25_obj is None:
        return []

    tokenized_query = query.lower().split()
    scores = bm25_obj.get_scores(tokenized_query)

    hits = []
    for idx, score in enumerate(scores):
        if score <= 0:
            continue
        
        chunk = chunks_list[idx]
        
        if category and chunk.meta.get("category") != category:
            continue
        if product and chunk.meta.get("product") != product:
            continue
        if audience and chunk.meta.get("audience") != audience:
            continue
        if status and chunk.meta.get("status") != status:
            continue

        hits.append(Hit(chunk=chunk, score=float(score)))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]
