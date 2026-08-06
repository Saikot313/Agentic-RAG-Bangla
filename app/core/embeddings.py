
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed_passages(texts: list[str]) -> np.ndarray:
    model = _get_model()
    prefixed = [f"passage: {t}" for t in texts]
    embeddings = model.encode(
        prefixed, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )
    return embeddings.astype("float32")


def embed_query(text: str) -> np.ndarray:
    model = _get_model()
    embedding = model.encode(
        [f"query: {text}"],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embedding.astype("float32")[0]
