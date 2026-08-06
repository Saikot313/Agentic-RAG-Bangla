from dataclasses import dataclass

from app.agents.query_agent import QueryUnderstanding, understand_query
from app.agents.retrieval_agent import RetrievedItem, retrieve
from app.agents.verifier_agent import VerificationResult, verify
from app.core.llm_client import get_llm_client

_ANSWER_SYSTEM_PROMPT = """You are a helpful document Q&A assistant that answers \
strictly using the provided CONTEXT. Rules:
- Answer in the same language as the QUESTION (Bangla question -> Bangla \
  answer, English question -> English answer). If asked for a bilingual \
  answer, give the Bangla answer first, then the English translation.
- Only use facts present in the CONTEXT. If the answer isn't in the CONTEXT, \
  say so honestly (in the appropriate language) instead of guessing.
- Be concise and directly answer the question; don't restate the whole \
  context.
"""


@dataclass
class PipelineResult:
    question: str
    detected_language: str
    rewritten_query: str
    answer: str
    retrieved: list[RetrievedItem]
    verification: VerificationResult
    retried: bool


def _build_context(retrieved: list[RetrievedItem]) -> str:
    parts = []
    for item in retrieved:
        parts.append(f"[Source: {item.chunk.source}]\n{item.chunk.text}")
    return "\n\n".join(parts)


def _generate_answer(question: str, context: str, bilingual: bool) -> str:
    llm = get_llm_client()
    user_prompt = (
        f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\n"
        f"{'Give a bilingual (Bangla + English) answer.' if bilingual else ''}"
    )
    return llm.generate(_ANSWER_SYSTEM_PROMPT, user_prompt, temperature=0.2)


def run_pipeline(
    question: str, embedding_dim: int, top_k: int = 5, bilingual: bool = False
) -> PipelineResult:
    # 1. Query Understanding Agent
    qu: QueryUnderstanding = understand_query(question)

    # 2. Retrieval Agent
    retrieved = retrieve(qu, embedding_dim=embedding_dim, top_k=top_k)
    context = _build_context(retrieved)

    # 3. Answer generation
    answer = _generate_answer(qu.rewritten, context, bilingual)

    # 4. Verifier / Fact-Check Agent
    verification = verify(answer, context)
    retried = False

    # 4b. If not grounded, retry once: broaden retrieval (more chunks) and
    # regenerate with an explicit instruction to stick to context.
    if not verification.is_grounded and top_k < 10:
        retried = True
        retrieved = retrieve(qu, embedding_dim=embedding_dim, top_k=top_k + 5)
        context = _build_context(retrieved)
        answer = _generate_answer(qu.rewritten, context, bilingual)
        verification = verify(answer, context)

    return PipelineResult(
        question=question,
        detected_language=qu.language,
        rewritten_query=qu.rewritten,
        answer=answer,
        retrieved=retrieved,
        verification=verification,
        retried=retried,
    )
