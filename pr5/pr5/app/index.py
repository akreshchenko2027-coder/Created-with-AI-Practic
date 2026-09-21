import os
import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
from rank_bm25 import BM25Okapi
from .documents import Chunk

INDEX_DIR = Path(os.getenv("INDEX_DIR", "pr5/data/index"))
DEFAULT_TOP_K = int(os.getenv("SEARCH_TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.0"))

@dataclass
class Hit:
    chunk: Chunk
    score: float

class Index:
    def __init__(self, chunks: List[Chunk], vectors: np.ndarray, model_name: str):
        self.chunks = chunks
        self.vectors = vectors
        self.model_name = model_name
        
        corpus = [chunk.text.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(corpus)

    def __len__(self) -> int:
        return len(self.chunks)

    def save(self, path: Path = INDEX_DIR):
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "index.pkl", "wb") as f:
            pickle.dump(self, f)

    def search_vector(
        self, 
        query_vector: np.ndarray, 
        top_k: int = DEFAULT_TOP_K, 
        threshold: float = SIMILARITY_THRESHOLD,
        category: Optional[str] = None,
        product: Optional[str] = None,
        audience: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Hit]:
        scores = np.dot(self.vectors, query_vector)
        top_indices = np.argsort(scores)[::-1]
        
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < threshold:
                continue
                
            chunk = self.chunks[idx]
            
            if category and chunk.meta.get("category") != category:
                continue
            if product and chunk.meta.get("product") != product:
                continue
            if audience and chunk.meta.get("audience") != audience:
                continue
            if status and chunk.meta.get("status") != status:
                continue

            results.append(Hit(chunk=chunk, score=score))
            if len(results) >= top_k:
                break
                
        return results

def save(idx: Index, path: Path = INDEX_DIR):
    idx.save(path)

def build(chunks: List[Chunk], vectors: np.ndarray, model_name: str) -> Index:
    idx = Index(chunks, vectors, model_name)
    idx.save()
    return idx

def load(path: Path = INDEX_DIR) -> Optional[Index]:
    index_file = path / "index.pkl"
    if not index_file.exists():
        return None
    with open(index_file, "rb") as f:
        return pickle.load(f)

def search(
    index: Index, 
    query_vector: np.ndarray, 
    top_k: int = DEFAULT_TOP_K, 
    threshold: float = SIMILARITY_THRESHOLD,
    category: Optional[str] = None,
    product: Optional[str] = None,
    audience: Optional[str] = None,
    status: Optional[str] = None
) -> List[Hit]:
    return index.search_vector(
        query_vector, 
        top_k=top_k, 
        threshold=threshold,
        category=category,
        product=product,
        audience=audience,
        status=status
    )
