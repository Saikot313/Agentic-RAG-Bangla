
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.document_loader import chunk_text, detect_language_ratio, split_sentences


def test_detect_language_bangla():
    text = "বাংলাদেশ দক্ষিণ এশিয়ার একটি দেশ। এর রাজধানী ঢাকা।"
    assert detect_language_ratio(text) == "bn"


def test_detect_language_english():
    text = "Bangladesh is a country in South Asia. Its capital is Dhaka."
    assert detect_language_ratio(text) == "en"


def test_detect_language_mixed():
    text = "Bangladesh (বাংলাদেশ) is a South Asian country."
    assert detect_language_ratio(text) == "mixed"


def test_split_sentences_bangla_daari():
    text = "এটি প্রথম বাক্য। এটি দ্বিতীয় বাক্য।"
    sentences = split_sentences(text)
    assert len(sentences) == 2


def test_chunk_text_respects_size():
    text = " ".join([f"Sentence number {i}." for i in range(100)])
    chunks = chunk_text(text, source="test.pdf", chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        # allow a little slack because we don't split mid-sentence
        assert len(c.text) < 300


def test_chunk_text_empty():
    chunks = chunk_text("", source="empty.pdf")
    assert chunks == []
