
import json
import os
from dataclasses import dataclass

import faiss
import numpy as np

from app.config import settings


@dataclass
class StoredChunk:
    text: str
    source: str
    chunk_id: int


class FaissVectorStore:
    def __init__(self, dim: int, index_dir: str | None = None):
        self.dim = dim
        self.index_dir = index_dir or settings.faiss_index_dir
        self.index = faiss.IndexFlatIP(dim)  # cosine sim via normalized dot product
        self.metadata: list[StoredChunk] = []
        os.makedirs(self.index_dir, exist_ok=True)
        self._maybe_load()

    def add(self, vectors: np.ndarray, chunks: list[StoredChunk]) -> None:
        assert len(vectors) == len(chunks)
        self.index.add(vectors)
        self.metadata.extend(chunks)
        self._persist()

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[tuple[StoredChunk, float]]:
        if self.index.ntotal == 0:
            return []
        scores, indices = self.index.search(query_vector.reshape(1, -1), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.metadata[idx], float(score)))
        return results

    def _index_path(self) -> str:
        return os.path.join(self.index_dir, "index.faiss")

    def _meta_path(self) -> str:
        return os.path.join(self.index_dir, "meta.json")

    def _persist(self) -> None:
        faiss.write_index(self.index, self._index_path())
        with open(self._meta_path(), "w", encoding="utf-8") as f:
            json.dump(
                [{"text": c.text, "source": c.source, "chunk_id": c.chunk_id} for c in self.metadata],
                f,
                ensure_ascii=False,
            )

    def _maybe_load(self) -> None:
        if os.path.exists(self._index_path()) and os.path.exists(self._meta_path()):
            self.index = faiss.read_index(self._index_path())
            with open(self._meta_path(), "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.metadata = [StoredChunk(**item) for item in raw]


_store: FaissVectorStore | None = None


def get_vector_store(dim: int) -> FaissVectorStore:
    global _store
    if _store is None:
        _store = FaissVectorStore(dim=dim)
    return _store
