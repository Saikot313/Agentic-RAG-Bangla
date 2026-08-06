from dataclasses import dataclass

from app.core.document_loader import detect_language_ratio
from app.core.llm_client import get_llm_client

_SYSTEM_PROMPT = """You are a query understanding module for a Bangla/English \
document search system. Given a user's question, produce a short JSON object \
with two fields:
- "rewritten": a clearer, more specific version of the query, in the SAME \
  language as the original question (Bangla stays Bangla, English stays \
  English).
- "translated": the query translated into the OTHER language (if original is \
  Bangla, translate to English; if English, translate to Bangla). This is \
  used to also search documents that are in the other language.

Respond with ONLY the JSON object, no other text.
"""


@dataclass
class QueryUnderstanding:
    original: str
    language: str  # "bn" | "en" | "mixed"
    rewritten: str
    translated: str  # query translated to the other language, for cross-lingual retrieval


def understand_query(question: str) -> QueryUnderstanding:
    language = detect_language_ratio(question)

    llm = get_llm_client()
    raw = llm.generate(_SYSTEM_PROMPT, question, temperature=0.0)

    rewritten, translated = question, question
    try:
        import json

        parsed = json.loads(raw)
        rewritten = parsed.get("rewritten", question)
        translated = parsed.get("translated", question)
    except (json.JSONDecodeError, AttributeError):
        # If the LLM didn't return clean JSON, fall back to using the
        # original question for both — retrieval still works, just without
        # the cross-lingual boost.
        pass

    return QueryUnderstanding(
        original=question, language=language, rewritten=rewritten, translated=translated
    )
