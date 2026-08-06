from typing import Literal, Optional

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    detected_languages: list[str]


class QueryRequest(BaseModel):
    question: str = Field(..., description="Question in Bangla, English, or mixed.")
    bilingual_answer: bool = Field(
        False, description="If true, return the answer in both Bangla and English."
    )
    top_k: Optional[int] = None


class RetrievedChunk(BaseModel):
    text: str
    source: str
    score: float


class VerificationResult(BaseModel):
    is_grounded: bool
    unsupported_claims: list[str]
    confidence: float


class QueryResponse(BaseModel):
    question: str
    detected_language: Literal["bn", "en", "mixed"]
    rewritten_query: str
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    verification: VerificationResult
    retried: bool
