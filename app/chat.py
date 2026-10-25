"""The chatbot. For each question:

guardrails → pick the policy (if a procedure is mentioned) → hybrid search of the policy PDFs →
the LLM answers from those passages + the doctor's note → code verifies every quote → answer with sources.

The LLM works out the policy's requirements itself from the passages it is given; nothing is hard-coded.
An LLM failure never crashes: the user gets a plain "try again" answer.
"""

import re
import time
import uuid

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from app import guardrails, llm, observability, policy_store
from app.intake import find_cpt, find_icd
from app.llm import LLMError
from app.models import ChatAnswer, LLMAnswer, PolicyQuote, RetrievalTrace
from app.prompts import ANSWER_PROMPT

NOTE_IN_QUERY_CHARS = 600  # how much of the note helps the search find the right policy passages


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


@traceable(name="hybrid_retrieve", run_type="retriever")
def hybrid_retrieve(retriever, query: str, policy_id: str | None, retrieval_mode: str) -> RetrievalTrace:
    # No score threshold: the reranker's order is useful, but its scores for broad questions ("does this meet
    # the policy?") are often negative even for the right passages. RAGAS context precision tracks relevance.
    trace = retriever.retrieve(query, policy_id, retrieval_mode)
    rt = get_current_run_tree()
    if rt:
        rt.add_metadata({"bm25": len(trace.bm25), "vector": len(trace.vector), "fused": len(trace.fused),
                         "kept": len(trace.reranked), "top_score": trace.reranked[0].score if trace.reranked else None})
    return trace


def passages_text(trace: RetrievalTrace) -> str:
    if not trace.reranked:
        return "(No policy passage matched this question.)"
    return "\n\n".join(f"[{h.source}, page {h.page}]\n{h.text}" for h in trace.reranked)


@traceable(name="verify_quotes", run_type="chain")
def verify_quotes(out: LLMAnswer, note: str, trace: RetrievalTrace) -> tuple[list[str], list[PolicyQuote], list[str]]:
    """Keep only quotes that are really in the note or the retrieved policy text; the file and page come from
    the passage the quote was found in, not from the LLM."""
    note_ok = [q for q in out.note_quotes if note and guardrails.quote_in_text(q, note)]
    policy_ok = []
    joined = " ".join(h.text for h in trace.reranked)  # a quote can run across two neighbouring chunks
    for q in out.policy_quotes:
        hit = next((h for h in trace.reranked if guardrails.quote_in_text(q.text, h.text)), None)
        if not hit and guardrails.quote_in_text(q.text, joined):
            start = " ".join(q.text.split()[:4])  # cite the passage where the quote starts
            hit = next((h for h in trace.reranked if guardrails.quote_in_text(start, h.text)), trace.reranked[0])
        if hit:
            policy_ok.append(PolicyQuote(text=q.text, source=hit.source, page=hit.page))
    unverified = [q for q in out.note_quotes if q not in note_ok] + \
                 [q.text for q in out.policy_quotes if q.text not in {p.text for p in policy_ok}]
    return note_ok, policy_ok, unverified


@traceable(name="answer_question", run_type="chain")
def _answer(question: str, note: str, cpt: str | None, icd: list[str], history: list[dict],
            retrieval_mode: str, retriever) -> ChatAnswer:
    timings: dict[str, float] = {}
    start = time.perf_counter()
    policy = policy_store.get_policy(cpt)
    code_check = policy_store.code_check(icd, cpt)
    if retriever is None:
        from app.retriever import get_retriever
        retriever = get_retriever()

    t = time.perf_counter()
    query = f"{question}\n{note[:NOTE_IN_QUERY_CHARS]}".strip()
    trace = hybrid_retrieve(retriever, query, policy.policy_id if policy else None, retrieval_mode)
    timings["retrieve"] = _ms(t)

    t = time.perf_counter()
    hist = "\n".join(f"{h.get('role', 'user')}: {h.get('content', '')}" for h in history[-6:])
    prompt = ANSWER_PROMPT.format(passages=passages_text(trace), code_check=code_check or "(no codes given)",
                                  note=note or "(no note provided)", history=hist or "(none)", question=question)
    out = llm.structured(prompt, LLMAnswer)
    timings["llm"] = _ms(t)

    t = time.perf_counter()
    note_ok, policy_ok, unverified = verify_quotes(out, note, trace)
    answer = guardrails.check_answer(re.sub(r"【[^】]*】", "", out.answer).strip())  # drop gpt-oss citation marks
    verdict = out.verdict
    if code_check and "NOT on the accepted list" in code_check and verdict == "meets":
        # The code list is exact data; never let the LLM call a note "meets" when the code isn't accepted.
        verdict = "does_not_meet"
        answer += "\n\nNote: the diagnosis code is not on Medicare's accepted list for this procedure."
    timings["verify"] = _ms(t)
    timings["total"] = _ms(start)

    rt = get_current_run_tree()
    if rt:
        rt.add_metadata({"verdict": verdict, "unverified_quotes": len(unverified)})
    return ChatAnswer(question=question, answer=answer, verdict=verdict, note_quotes=note_ok, policy_quotes=policy_ok,
                      unverified_quotes=unverified, policy=policy, code_check=code_check, retrieval=trace,
                      timings_ms=timings)


def answer(question: str, note: str = "", cpt: str | None = None, icd_codes: list[str] | None = None,
           history: list[dict] | None = None, retrieval_mode: str = "full", retriever=None,
           mode: str = "chat") -> ChatAnswer:
    """Answer one question. Raises GuardrailError for input we refuse; never raises on LLM errors."""
    guardrails.check_chat_message(question)
    question, removed_q = guardrails.redact_phi(guardrails.clean_text(question))
    note, removed_n = guardrails.redact_phi(guardrails.clean_text(note)) if note.strip() else ("", [])
    if note:
        guardrails.check_note(note)
    procs = policy_store.procedures()
    cpt = cpt or find_cpt(question, procs)
    icd = icd_codes or find_icd(question)

    run_id = str(uuid.uuid4()) if observability.TRACING_ENABLED else None
    extra = {"run_id": run_id, "tags": [f"mode:{mode}", f"procedure:{cpt or 'any'}"],
             "metadata": {"mode": mode, "cpt_code": cpt, "retrieval_mode": retrieval_mode}}
    try:
        result = _answer(question, note, cpt, icd, history or [], retrieval_mode, retriever, langsmith_extra=extra)
    except LLMError as e:
        result = ChatAnswer(question=question, verdict="not_applicable",
                            answer="Sorry, I couldn't reach the AI service just now (it may be busy, out of its daily "
                                   f"limit, or offline). Please try again in a minute. ({str(e)[:120]})")
    result.run_id = run_id
    result.removed_personal_data = sorted(set(removed_q + removed_n))
    return result
