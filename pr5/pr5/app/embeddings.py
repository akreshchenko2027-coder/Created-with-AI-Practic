import os
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def embed_passages(texts: list[str]) -> np.ndarray:
    model = get_model()
    if "e5" in MODEL_NAME.lower():
        formatted_texts = [f"passage: {t}" for t in texts]
    else:
        formatted_texts = texts
        
    embeddings = model.encode(formatted_texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings

def embed_query(query: str) -> np.ndarray:
    model = get_model()
    if "e5" in MODEL_NAME.lower():
        formatted_query = f"query: {query}"
    else:
        formatted_query = query
        
    embedding = model.encode(formatted_query, convert_to_numpy=True, normalize_embeddings=True)
    return embedding
