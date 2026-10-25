"""Conversation about one case. Normal questions are answered from the case data only.
"What if ..." questions become one added note line, and the case is re-reviewed on a COPY so the
reviewer sees a before/after diff. The saved case only changes if the reviewer clicks "Add to case".
"""

import json
import re
import uuid

from langsmith import traceable

from app import graph_store, guardrails, llm, observability
from app.llm import LLMError
from app.models import CaseRequest, CaseReview, WhatIfNote
from app.prompts import CHAT_PROMPT, WHAT_IF_PROMPT

WHAT_IF = re.compile(r"^\s*(what if|what would happen if|suppose|imagine|if the patient)\b", re.I)


def is_what_if(message: str) -> bool:
    return bool(WHAT_IF.search(message))


def case_context(case: CaseRequest, review: CaseReview | None) -> str:
    """Compact JSON of everything the chat may use: the case, the results with quotes, and the policy path."""
    data = {"case": case.model_dump(), "policy_path": graph_store.explain_path(case.cpt_code)}
    if review:
        data["decision"] = review.decision.model_dump()
        data["criteria"] = [r.model_dump(include={"criterion_id", "criterion_text", "status", "reasoning",
                                                  "note_evidence", "policy_evidence", "confidence"})
                            for r in review.results]
        data["groups"] = [g.model_dump() for g in review.group_statuses]
    return json.dumps(data, indent=1)


def decision_diff(before: CaseReview, after: CaseReview) -> dict:
    old = {r.criterion_id: r.status for r in before.results}
    changed = [{"criterion_id": r.criterion_id, "before": old.get(r.criterion_id), "after": r.status}
               for r in after.results if old.get(r.criterion_id) != r.status]
    return {"before": before.decision.outcome, "after": after.decision.outcome, "changed": changed}


@traceable(name="what_if", run_type="chain")
def what_if(case: CaseRequest, review: CaseReview, message: str) -> dict:
    from app.pipeline import review_case  # imported here to avoid a circular import

    note = llm.structured(WHAT_IF_PROMPT.format(question=message), WhatIfNote).note_line.strip()
    copy = case.model_copy(update={"clinical_notes": f"{case.clinical_notes}\n{note}"})
    new = review_case(copy, mode="what_if", save=False)
    diff = decision_diff(review, new)
    failed = [r for r in new.decision.reasons if r.startswith("Automatic check failed")]
    if failed:  # don't present a failed re-check as a real "what if" outcome
        return {"answer": f"I couldn't re-check the case with \"{note}\" ({failed[0][:160]}). Please try again "
                          "in a minute (the LLM may be rate-limited).", "what_if": None}
    changes = ", ".join(f"{c['criterion_id']}: {c['before']} → {c['after']}" for c in diff["changed"]) or "no criteria"
    answer = (f"If the notes said: \"{note}\" the decision would be {diff['after']} (now {diff['before']}). "
              f"Changed: {changes}. The saved case is unchanged unless you click \"Add to case\".")
    return {"answer": answer, "what_if": {"note_line": note, **diff, "after_review": new.model_dump()}}


@traceable(name="chat", run_type="chain")
def _answer(case: CaseRequest, review: CaseReview | None, message: str, history: list[dict]) -> dict:
    if is_what_if(message):
        if review is None:
            return {"answer": "Run the review first, then ask what-if questions.", "what_if": None}
        return what_if(case, review, message)
    hist = "\n".join(f"{h.get('role', 'user')}: {h.get('content', '')}" for h in history[-6:])
    prompt = CHAT_PROMPT.format(case_json=case_context(case, review), history=hist, question=message)
    # gpt-oss sometimes adds internal citation marks like 【2a.policy_evidence[0]】; drop them.
    answer = re.sub(r"【[^】]*】", "", llm.text(prompt)).strip()
    return {"answer": guardrails.check_answer(answer), "what_if": None}


def answer(case: CaseRequest, review: CaseReview | None, message: str, history: list[dict] | None = None) -> dict:
    """Returns {answer, what_if, run_id}. Raises GuardrailError for rejected messages; never raises on LLM errors."""
    guardrails.check_chat_message(message)
    mode = "what_if" if is_what_if(message) else "chat"
    run_id = str(uuid.uuid4()) if observability.TRACING_ENABLED else None
    try:
        out = _answer(case, review, message, history or [], langsmith_extra={
            "run_id": run_id, "tags": [f"mode:{mode}"], "metadata": {"case_id": case.case_id, "mode": mode}})
    except LLMError as e:
        out = {"answer": f"Sorry, the assistant could not answer right now ({str(e)[:120]}).", "what_if": None}
    return {**out, "run_id": run_id, "mode": mode}
