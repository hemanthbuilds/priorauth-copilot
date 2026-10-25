"""The LLM judges messy text. This file applies the fixed business rules, so the same results always
give the same decision. This makes it consistent, testable, and auditable.

The AI never denies a request: anything that might be a "no" goes to a human clinician.
"""

from app.models import CriterionResult, Decision, GroupStatus, Status


def group_status(group_logic: str, statuses: list[Status]) -> Status:
    """Combine the statuses of the criteria inside one group."""
    if not statuses:
        return "missing"  # an empty group proves nothing (policy_store also rejects empty groups)
    if group_logic == "ALL":
        if all(s == "met" for s in statuses):
            return "met"
        if any(s == "not_met" for s in statuses):
            return "not_met"  # one failed requirement means the group can never be met
        return "missing"
    if group_logic == "ANY":
        if any(s == "met" for s in statuses):
            return "met"
        if all(s == "not_met" for s in statuses):
            return "not_met"  # every option has failed
        return "missing"
    raise ValueError(f"unknown group logic: {group_logic}")


def decide(group_statuses: list[GroupStatus], results: list[CriterionResult],
           diagnosis_supported: bool) -> Decision:
    """Turn group statuses into APPROVE / REQUEST_INFO / SEND_TO_CLINICIAN. Rules run in this order."""
    by_id = {r.criterion_id: r for r in results}

    # 1. Low-confidence checks or an unsupported diagnosis need a human clinician. A plain unsure "missing" is
    #    not escalated: asking the doctor for that item (rule 3) is the right step. Missing results that a
    #    guardrail flagged (fake quote, no policy text, AI failure) still go to a clinician.
    low = [r for r in results if r.confidence == "low" and (r.status != "missing" or r.flags)
           and not _settled_without(r, group_statuses, by_id)]
    if low or not diagnosis_supported:
        reasons = []
        if not diagnosis_supported:
            reasons.append("The diagnosis codes do not support this procedure under the policy.")
        for r in low:
            reasons.append(f"Criterion {r.criterion_id} could not be checked with confidence ({r.status}).")
        return Decision(outcome="SEND_TO_CLINICIAN", reasons=reasons, missing_items=_missing_items(group_statuses, by_id))

    # 2. A group that can no longer be satisfied → a clinician decides (the AI never denies).
    failed = [g for g in group_statuses if g.status == "not_met"]
    if failed:
        reasons = [f"Group {g.group_id} ({g.description}) is not met." for g in failed]
        return Decision(outcome="SEND_TO_CLINICIAN", reasons=reasons, missing_items=[])

    # 3. Something is undocumented → ask the doctor for it.
    open_groups = [g for g in group_statuses if g.status == "missing"]
    if open_groups:
        reasons = [f"Group {g.group_id} ({g.description}) needs more information." for g in open_groups]
        return Decision(outcome="REQUEST_INFO", reasons=reasons, missing_items=_missing_items(group_statuses, by_id))

    # 4. Everything is met.
    return Decision(outcome="APPROVE", reasons=[f"Group {g.group_id} ({g.description}) is met." for g in group_statuses],
                    missing_items=[])


def _settled_without(r: CriterionResult, group_statuses: list[GroupStatus], by_id: dict[str, CriterionResult]) -> bool:
    """True if r can't change the outcome: it sits in an ANY group that another criterion already meets
    with confidence (e.g. sleep apnea is clearly documented, so an unsure 'parasomnia' answer doesn't matter)."""
    for g in group_statuses:
        if r.criterion_id in g.criterion_ids and g.logic == "ANY":
            others = [by_id[c] for c in g.criterion_ids if c != r.criterion_id and c in by_id]
            return any(o.status == "met" and o.confidence != "low" for o in others)
    return False


def _missing_items(group_statuses: list[GroupStatus], by_id: dict[str, CriterionResult]) -> list[str]:
    """For every group still open, list the criteria the doctor could document to close it."""
    items = []
    for g in group_statuses:
        if g.status != "missing":
            continue
        missing = [by_id[c] for c in g.criterion_ids if c in by_id and by_id[c].status == "missing"]
        prefix = "One of: " if g.logic == "ANY" and len(missing) > 1 else ""
        for r in missing:
            items.append(f"{prefix}{r.criterion_id}: {r.criterion_text}")
    return items


def fallback_decision(reason: str) -> Decision:
    """Used when an automatic step fails: always hand the case to a human."""
    return Decision(outcome="SEND_TO_CLINICIAN", reasons=[f"Automatic check failed: {reason}"], missing_items=[])
