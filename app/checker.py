"""Checks ONE policy criterion against the notes with the LLM, using the policy text found by retrieval.

The LLM only says met / not_met / missing with quotes. Code then attaches the exact policy source and
page, and guardrails (in the pipeline) verify the note quote is real.
"""

from langsmith import traceable

from app import llm
from app.guardrails import quote_in_text
from app.models import CaseFacts, CaseRequest, CheckOutput, Criterion, CriterionResult, Evidence, Hit
from app.prompts import CHECK_PROMPT


def policy_context(chunks: list[Hit]) -> str:
    return "\n\n".join(f"[{c.source}, page {c.page}]\n{c.text}" for c in chunks)


def _first_words(text: str, n: int = 30) -> str:
    words = text.split()
    return " ".join(words[:n]) + (" ..." if len(words) > n else "")


def policy_evidence(quote: str | None, chunks: list[Hit]) -> list[Evidence]:
    """Use the LLM's policy quote only if it is really in a retrieved chunk; otherwise show the best chunk."""
    if quote:
        for c in chunks:
            if quote_in_text(quote, c.text):
                return [Evidence(source=c.source, page=c.page, quote=_first_words(quote))]
    if chunks:
        return [Evidence(source=chunks[0].source, page=chunks[0].page, quote=_first_words(chunks[0].text))]
    return []


@traceable(name="llm_check", run_type="chain")
def llm_check(criterion: Criterion, case: CaseRequest, facts: CaseFacts, chunks: list[Hit]) -> CheckOutput:
    prompt = CHECK_PROMPT.format(criterion_text=criterion.text, policy_context=policy_context(chunks),
                                 facts=facts.model_dump_json(), notes=case.clinical_notes)
    return llm.structured(prompt, CheckOutput)


def check_criterion(criterion: Criterion, case: CaseRequest, facts: CaseFacts,
                    chunks: list[Hit]) -> CriterionResult:
    """chunks = the top reranked policy chunks for this criterion."""
    out = llm_check(criterion, case, facts, chunks)
    return CriterionResult(
        criterion_id=criterion.id, criterion_text=criterion.text, status=out.status, reasoning=out.reasoning,
        note_evidence=out.note_evidence, policy_evidence=policy_evidence(out.policy_quote, chunks),
        confidence=out.confidence)
