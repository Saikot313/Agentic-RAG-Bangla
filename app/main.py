import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agents.orchestrator import run_pipeline
from app.config import settings
from app.core.document_loader import detect_language_ratio, load_and_chunk_pdf
from app.core.embeddings import embed_passages
from app.core.vector_store import StoredChunk, get_vector_store
from app.schemas import (
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    UploadResponse,
    VerificationResult,
)

app = FastAPI(
    title="Agentic RAG for Bangla Document Intelligence",
    description="Multi-agent RAG pipeline with bilingual (Bangla/English) support.",
    version="0.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.on_event("startup")
def preload_embedding_model():
    from app.core.embeddings import _get_model

    _get_model()


@app.get("/")
def serve_ui():
    return FileResponse(_STATIC_DIR / "index.html")


_EMBEDDING_DIM = 768


@app.get("/health")
def health():
    return {"status": "ok", "llm_provider": settings.llm_provider}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported right now.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        chunks = load_and_chunk_pdf(
            tmp_path,
            source_name=file.filename,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Could not extract any text from this PDF (it may be scanned/image-only).",
            )

        texts = [c.text for c in chunks]
        vectors = embed_passages(texts)

        store = get_vector_store(dim=_EMBEDDING_DIM)
        stored_chunks = [
            StoredChunk(text=c.text, source=c.source, chunk_id=c.chunk_id) for c in chunks
        ]
        store.add(vectors, stored_chunks)

        languages = sorted({detect_language_ratio(t) for t in texts})

        return UploadResponse(
            filename=file.filename, chunks_indexed=len(chunks), detected_languages=languages
        )
    finally:
        os.unlink(tmp_path)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    top_k = request.top_k or settings.top_k
    result = run_pipeline(
        question=request.question,
        embedding_dim=_EMBEDDING_DIM,
        top_k=top_k,
        bilingual=request.bilingual_answer,
    )

    return QueryResponse(
        question=result.question,
        detected_language=result.detected_language,
        rewritten_query=result.rewritten_query,
        answer=result.answer,
        retrieved_chunks=[
            RetrievedChunk(text=item.chunk.text, source=item.chunk.source, score=item.score)
            for item in result.retrieved
        ],
        verification=VerificationResult(
            is_grounded=result.verification.is_grounded,
            unsupported_claims=result.verification.unsupported_claims,
            confidence=result.verification.confidence,
        ),
        retried=result.retried,
    )