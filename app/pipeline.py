"""review_case(): runs one prior authorization review end to end and times every stage.

guardrails → extract facts (LLM) → policy lookup (exact) → per criterion, in parallel:
hybrid retrieval + LLM check + quote check → group logic + decision (code) → save.

Any LLM failure falls back to SEND_TO_CLINICIAN with a clear reason; it never crashes.
"""

import time
import uuid

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from langsmith.utils import ContextThreadPoolExecutor

from app import config, db, guardrails, observability, policy_store
from app.checker import check_criterion
from app.decision import decide, fallback_decision, group_status
from app.extractor import extract_facts
from app.llm import LLMError
from app.models import CaseFacts, CaseRequest, CaseReview, Criterion, CriterionResult, GroupStatus, RetrievalTrace


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


@traceable(name="policy_lookup", run_type="chain")
def policy_lookup(case: CaseRequest) -> dict:
    groups = policy_store.get_criteria(case.cpt_code)
    return {"policy": policy_store.get_policy(case.cpt_code), "groups": groups,
            "diagnosis_supported": policy_store.diagnosis_supports(case.icd_codes, case.cpt_code),
            "counts": {"groups": len(groups), "criteria": sum(len(g.criteria) for g in groups)}}


@traceable(name="hybrid_retrieve", run_type="retriever")
def hybrid_retrieve(retriever, query: str, policy_id: str, retrieval_mode: str) -> RetrievalTrace:
    trace = retriever.retrieve(query, policy_id, retrieval_mode)
    rt = get_current_run_tree()
    if rt:
        rt.add_metadata({"bm25": len(trace.bm25), "vector": len(trace.vector), "fused": len(trace.fused),
                         "reranked": len(trace.reranked),
                         "top_score": trace.reranked[0].score if trace.reranked else None})
    return trace


@traceable(name="check_criterion", run_type="chain")
def check_one(criterion: Criterion, case: CaseRequest, facts: CaseFacts, retriever, policy_id: str,
              retrieval_mode: str) -> tuple[CriterionResult, RetrievalTrace, dict, str | None]:
    """Retrieve + check + quote-check one criterion. Returns (result, trace, timings, error)."""
    t = time.perf_counter()
    trace = hybrid_retrieve(retriever, criterion.text, policy_id, retrieval_mode)
    timings = {"retrieve": _ms(t)}

    top = trace.reranked[0].score if trace.reranked else None
    uses_reranker = retrieval_mode != "vector_only"  # vector-only scores are cosine, not rerank scores
    if not trace.reranked or (uses_reranker and top < config.MIN_RERANK_SCORE):
        return guardrails.low_retrieval_result(criterion.id, criterion.text, top), trace, timings, None

    t = time.perf_counter()
    try:
        result = check_criterion(criterion, case, facts, trace.reranked)
        error = None
    except LLMError as e:
        result = CriterionResult(criterion_id=criterion.id, criterion_text=criterion.text, status="missing",
                                 confidence="low", reasoning="Automatic check failed.", note_evidence=None,
                                 policy_evidence=[], flags=[str(e)[:300]])
        error = f"criterion {criterion.id}: {e}"
    timings["llm_check"] = _ms(t)
    return guardrails.apply_quote_check(result, case.clinical_notes), trace, timings, error


@traceable(name="decide", run_type="chain")
def decide_step(groups, results: list[CriterionResult], diagnosis_supported: bool):
    by_id = {r.criterion_id: r.status for r in results}
    statuses = [GroupStatus(group_id=g.group_id, logic=g.logic, description=g.description,
                            criterion_ids=[c.id for c in g.criteria],
                            status=group_status(g.logic, [by_id[c.id] for c in g.criteria]))
                for g in groups]
    return statuses, decide(statuses, results, diagnosis_supported)


@traceable(name="guardrails", run_type="chain")
def guardrail_summary(flags: list[str]) -> list[str]:
    return flags  # traced so all flags are visible in one place in LangSmith


@traceable(name="review_case", run_type="chain")
def _review(case: CaseRequest, mode: str, retrieval_mode: str, retriever) -> CaseReview:
    timings: dict[str, float] = {}
    start = time.perf_counter()
    flags = guardrails.check_case(case, policy_store.supported_cpts())  # raises GuardrailError (no LLM call)

    t = time.perf_counter()
    try:
        facts = extract_facts(case)
    except LLMError as e:
        timings["extract_facts"] = timings["total"] = _ms(t)
        return CaseReview(case_id=case.case_id, mode=mode, decision=fallback_decision(f"fact extraction: {e}"),
                          guardrail_flags=flags + [str(e)[:300]], timings_ms=timings)
    timings["extract_facts"] = _ms(t)

    t = time.perf_counter()
    lookup = policy_lookup(case)
    timings["policy_lookup"] = _ms(t)
    criteria = [c for g in lookup["groups"] for c in g.criteria]

    if retriever is None:
        from app.retriever import get_retriever
        retriever = get_retriever()

    t = time.perf_counter()
    # ContextThreadPoolExecutor keeps each thread's work inside this trace in LangSmith.
    # 2 at a time keeps a review under the free Groq tier's 8,000 tokens-per-minute limit.
    with ContextThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(check_one, c, case, facts, retriever, lookup["policy"].policy_id, retrieval_mode,
                               langsmith_extra={"name": f"check_criterion: {c.id}"}) for c in criteria]
        outputs = [f.result() for f in futures]
    timings["criteria_wall"] = _ms(t)
    timings["retrieve"] = round(sum(o[2].get("retrieve", 0) for o in outputs), 1)
    timings["llm_check"] = round(sum(o[2].get("llm_check", 0) for o in outputs), 1)

    results = [o[0] for o in outputs]
    errors = [o[3] for o in outputs if o[3]]
    for r in results:
        flags.extend(f"{r.criterion_id}: {f}" for f in r.flags)

    t = time.perf_counter()
    group_statuses, decision = decide_step(lookup["groups"], results, lookup["diagnosis_supported"])
    if errors:
        decision = fallback_decision("; ".join(errors)[:500])
    timings["decide"] = _ms(t)
    guardrail_summary(flags)
    timings["total"] = _ms(start)

    rt = get_current_run_tree()
    if rt:
        rt.add_metadata({"outcome": decision.outcome})
        rt.add_tags([f"outcome:{decision.outcome}"])
    return CaseReview(
        case_id=case.case_id, mode=mode, facts=facts, policy=lookup["policy"],
        diagnosis_supported=lookup["diagnosis_supported"], results=results, group_statuses=group_statuses,
        decision=decision, retrieval={c.id: o[1] for c, o in zip(criteria, outputs)}, guardrail_flags=flags,
        timings_ms=timings)


def review_case(case: CaseRequest, mode: str = "review", retrieval_mode: str = "full", retriever=None,
                save: bool = True) -> CaseReview:
    """Run a review. mode tags the trace (review | what_if | eval); retrieval_mode is for the ablation."""
    run_id = str(uuid.uuid4()) if observability.TRACING_ENABLED else None
    extra = {"run_id": run_id, "tags": [f"mode:{mode}", f"procedure:{case.cpt_code}"],
             "metadata": {"case_id": case.case_id, "cpt_code": case.cpt_code, "mode": mode,
                          "retrieval_mode": retrieval_mode}}
    review = _review(case, mode, retrieval_mode, retriever, langsmith_extra=extra)
    review.run_id = run_id
    if save:
        review.review_id = db.save_review(review)
        db.set_status(case.case_id, "Reviewed")
        db.audit(case.case_id, "system", "reviewed", {"outcome": review.decision.outcome, "mode": mode})
    return review
