
from dataclasses import dataclass

from app.agents.query_agent import QueryUnderstanding
from app.core.embeddings import embed_query
from app.core.vector_store import StoredChunk, get_vector_store


@dataclass
class RetrievedItem:
    chunk: StoredChunk
    score: float


def retrieve(qu: QueryUnderstanding, embedding_dim: int, top_k: int = 5) -> list[RetrievedItem]:
    store = get_vector_store(dim=embedding_dim)

    candidates: dict[str, RetrievedItem] = {}

    for query_text in {qu.rewritten, qu.translated}:
        vec = embed_query(query_text)
        for chunk, score in store.search(vec, top_k=top_k):
            key = f"{chunk.source}::{chunk.chunk_id}"
            # keep the higher score if this chunk was retrieved by both query variants
            if key not in candidates or score > candidates[key].score:
                candidates[key] = RetrievedItem(chunk=chunk, score=score)

    ranked = sorted(candidates.values(), key=lambda item: item.score, reverse=True)
    return ranked[:top_k]
