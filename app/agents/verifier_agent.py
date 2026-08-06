
import json
from dataclasses import dataclass

from app.core.llm_client import get_llm_client

_SYSTEM_PROMPT = """You are a strict fact-checking module. You will be given \
a CONTEXT (retrieved document excerpts) and an ANSWER that was generated from \
that context. Check whether every factual claim in the ANSWER is directly \
supported by the CONTEXT.

Respond with ONLY a JSON object:
{
  "is_grounded": true/false,
  "unsupported_claims": ["claim 1 not supported", ...],
  "confidence": 0.0-1.0
}

If the answer says it doesn't know / isn't in the document, that always \
counts as grounded (true). Be strict: an unsupported claim is any statement \
of fact not present in the context, even if it's plausible or general \
knowledge.
"""


@dataclass
class VerificationResult:
    is_grounded: bool
    unsupported_claims: list[str]
    confidence: float


def verify(answer: str, context: str) -> VerificationResult:
    llm = get_llm_client()
    user_prompt = f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"
    raw = llm.generate(_SYSTEM_PROMPT, user_prompt, temperature=0.0)

    try:
        parsed = json.loads(raw)
        return VerificationResult(
            is_grounded=bool(parsed.get("is_grounded", False)),
            unsupported_claims=list(parsed.get("unsupported_claims", [])),
            confidence=float(parsed.get("confidence", 0.5)),
        )
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
        # Fail safe: if we can't parse the verifier's output, don't silently
        # trust the answer — flag it as unverified with low confidence.
        return VerificationResult(
            is_grounded=False,
            unsupported_claims=["verifier_parse_error"],
            confidence=0.0,
        )
