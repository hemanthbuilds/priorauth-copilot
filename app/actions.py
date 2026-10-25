"""What happens after the reviewer decides: draft a letter, let them edit it, then finalize.

confirm  → reviewer agrees with the AI outcome; a draft is returned.
override → reviewer picks another outcome (with a reason); a draft is returned.
finalize → the (edited) letter is saved, the case status changes, and the audit log records it.
Nothing is ever sent anywhere: this is a proof of concept.
"""

import json

from app import db, llm, observability
from app.llm import LLMError
from app.models import CaseRequest, CaseReview
from app.prompts import LETTER_PROMPT

LETTER_TYPES = {"APPROVE": "approval", "REQUEST_INFO": "info_request", "SEND_TO_CLINICIAN": "clinician_handoff"}


class ActionError(Exception):
    """The request does not make sense in the case's current state."""


def template_letter(case: CaseRequest, review: CaseReview, outcome: str) -> str:
    """Plain fallback used when the LLM is unavailable, so the reviewer can always continue."""
    policy = f"{review.policy.title}" if review.policy else "the applicable policy"
    if outcome == "APPROVE":
        body = f"The request for CPT {case.cpt_code} has been reviewed against {policy} and is approved."
    elif outcome == "REQUEST_INFO":
        items = "\n".join(f"- {m}" for m in review.decision.missing_items) or "- (reviewer to list items)"
        body = f"To complete the review of CPT {case.cpt_code} under {policy}, please send:\n{items}"
    else:
        body = (f"Internal handoff: case {case.case_id} (CPT {case.cpt_code}) needs clinical review.\n"
                + "\n".join(f"- {r}" for r in review.decision.reasons))
    return f"Re: {case.case_id} ({case.patient_name})\n\n{body}\n\nPrior Authorization Review Team"


def draft_letter(case: CaseRequest, review: CaseReview, outcome: str) -> str:
    prompt = LETTER_PROMPT.format(
        letter_type=LETTER_TYPES[outcome], case_json=case.model_dump_json(),
        decision_json=json.dumps({**review.decision.model_dump(), "outcome": outcome}),
        policy_json=review.policy.model_dump_json() if review.policy else "{}")
    try:
        return llm.text(prompt)
    except LLMError:
        return template_letter(case, review, outcome)


def handle_action(case_id: str, action: str, outcome: str | None = None, reason: str | None = None,
                  letter_text: str | None = None, actor: str = "reviewer") -> dict:
    found = db.get_case(case_id)
    latest = db.latest_review(case_id)
    if not found or not latest:
        raise ActionError("Review the case before taking an action.")
    case, _ = found
    review: CaseReview = latest["review"]
    ai_outcome = review.decision.outcome

    if action in ("confirm", "override"):
        if action == "override":
            if outcome not in LETTER_TYPES or not (reason or "").strip():
                raise ActionError("Override needs an outcome and a reason.")
            final = outcome
        else:
            final = ai_outcome
        draft = draft_letter(case, review, final)
        db.save_action(case_id, latest["review_id"], action, ai_outcome, final, reason, draft, actor)
        db.audit(case_id, actor, "confirmed" if action == "confirm" else "overridden",
                 {"ai_outcome": ai_outcome, "final_outcome": final, "reason": reason})
        # Track how often humans agree with the AI.
        observability.add_feedback(latest["run_id"], "human_agreed", 1 if final == ai_outcome else 0, reason)
        return {"status": "draft", "final_outcome": final, "letter_text": draft}

    if action == "finalize":
        prev = db.latest_action(case_id)
        if not prev or prev["action"] not in ("confirm", "override"):
            raise ActionError("Confirm or override the decision before finalizing.")
        text = (letter_text or prev["letter_text"] or "").strip()
        if not text:
            raise ActionError("The letter is empty.")
        final = prev["final_outcome"]
        db.save_action(case_id, latest["review_id"], "finalize", ai_outcome, final, prev["reason"], text, actor,
                       finalized=True)
        status = db.STATUS_FOR_OUTCOME[final]
        db.set_status(case_id, status)
        db.audit(case_id, actor, "finalized", {"final_outcome": final, "status": status,
                                               "edited": text != (prev["letter_text"] or "").strip()})
        return {"status": status, "final_outcome": final, "letter_text": text}

    raise ActionError(f"Unknown action: {action}")
