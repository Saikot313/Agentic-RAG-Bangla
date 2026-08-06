
import re
from dataclasses import dataclass

from pypdf import PdfReader

# Bangla Unicode block: U+0980–U+09FF
_BANGLA_RE = re.compile(r"[\u0980-\u09FF]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[।.!?])\s+")


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


def detect_language_ratio(text: str) -> str:
    """Rough language tag for a piece of text: 'bn', 'en', or 'mixed'."""
    if not text.strip():
        return "en"
    bangla_chars = len(_BANGLA_RE.findall(text))
    letter_chars = sum(c.isalpha() for c in text)
    if letter_chars == 0:
        return "en"
    ratio = bangla_chars / letter_chars
    if ratio > 0.6:
        return "bn"
    if ratio < 0.1:
        return "en"
    return "mixed"


def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str, source: str, chunk_size: int = 800, chunk_overlap: int = 120
) -> list[Chunk]:
    """Sentence-aware sliding window chunking (works for Bangla + English)."""
    sentences = split_sentences(text)
    chunks: list[Chunk] = []
    current = ""
    chunk_id = 0

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) > chunk_size and current:
            chunks.append(Chunk(text=current, source=source, chunk_id=chunk_id))
            chunk_id += 1
            # start next chunk with overlap (tail of previous chunk)
            overlap_text = current[-chunk_overlap:] if chunk_overlap else ""
            current = f"{overlap_text} {sentence}".strip()
        else:
            current = candidate

    if current:
        chunks.append(Chunk(text=current, source=source, chunk_id=chunk_id))

    return chunks


def load_and_chunk_pdf(
    path: str, source_name: str, chunk_size: int = 800, chunk_overlap: int = 120
) -> list[Chunk]:
    raw_text = extract_text_from_pdf(path)
    return chunk_text(raw_text, source_name, chunk_size, chunk_overlap)
