"""Background quality scoring with RAGAS after every review and chat answer. Runs in a FastAPI
BackgroundTask, so it never slows the reviewer down. Scores go to SQLite (eval_scores) and, when
tracing is on, to the LangSmith trace as feedback. If scoring fails, a null score is saved and logged.

Metrics (no ground truth needed):
- faithfulness: is each criterion's reasoning supported by its context (retrieved policy text + the notes)?
- context_precision: are the reranked policy chunks actually relevant to the criterion?
- answer_relevancy: does a chat answer address the question?
"""

import asyncio
import logging
import threading

from app import config, db, observability
from app.models import CaseReview

log = logging.getLogger("live_eval")
CASE_LEVEL = "_case"  # criterion_id used for the per-case average rows
RATE_LIMIT_TRIES, RATE_LIMIT_WAIT_S = 4, 20
_SCORING_LOCK = threading.Lock()


def _fastembed_for_ragas():
    """Adapter so RAGAS can use our local fastembed model (no OpenAI key needed)."""
    from ragas.embeddings.base import BaseRagasEmbedding

    from app.retriever import get_embedder

    class FastEmbedRagas(BaseRagasEmbedding):
        def embed_text(self, text: str, **kwargs) -> list[float]:
            return next(iter(get_embedder().embed([text]))).tolist()

        async def aembed_text(self, text: str, **kwargs) -> list[float]:
            return self.embed_text(text)

    return FastEmbedRagas()


def metrics() -> dict:
    """Build the RAGAS metrics. RAGAS talks to Groq through its OpenAI-compatible endpoint.
    Built fresh inside each event loop, because the async HTTP client can't be shared across loops."""
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (AnswerRelevancy, ContextPrecisionWithoutReference, ContextRecall,
                                          Faithfulness)

    client = AsyncOpenAI(api_key=config.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1", max_retries=3)
    # Reasoning models (gpt-oss) spend output tokens on thinking, so allow room for it.
    extra = {"reasoning_effort": "low"} if config.RAGAS_MODEL.startswith("openai/gpt-oss") else {}
    judge = llm_factory(config.RAGAS_MODEL, provider="openai", client=client, temperature=0, max_tokens=4096, **extra)
    return {"faithfulness": Faithfulness(llm=judge),
            "context_precision": ContextPrecisionWithoutReference(llm=judge),
            "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=_fastembed_for_ragas()),
            "context_recall": ContextRecall(llm=judge)}  # needs a reference, so only the offline eval uses it


async def _run_jobs(jobs: list[tuple[str, dict]]) -> list[float | None]:
    try:
        built = metrics()
    except Exception as e:
        log.warning("RAGAS setup failed: %s", e)
        return [None] * len(jobs)
    out = []
    for metric, inputs in jobs:  # one at a time, to stay inside Groq rate limits
        value = None
        for attempt in range(RATE_LIMIT_TRIES):
            try:
                result = await built[metric].ascore(**inputs)
                value = None if result.value is None else round(float(result.value), 3)
                break
            except Exception as e:
                # Free Groq tiers allow few tokens per minute: wait and retry rate limits, give up on anything else.
                if "rate_limit" in str(e) and attempt < RATE_LIMIT_TRIES - 1:
                    await asyncio.sleep(RATE_LIMIT_WAIT_S)
                    continue
                log.warning("RAGAS %s failed: %s", metric, e)
                break
        out.append(value)
    return out


def score_many(jobs: list[tuple[str, dict]]) -> list[float | None]:
    """Score (metric, inputs) pairs. Each failure (e.g. a rate limit) becomes None and is logged.
    The lock makes background jobs from several reviews take turns instead of fighting over the rate limit."""
    if not jobs:
        return []
    with _SCORING_LOCK:
        return asyncio.run(_run_jobs(jobs))


def _avg(values: list[float | None]) -> float | None:
    real = [v for v in values if v is not None]
    return round(sum(real) / len(real), 3) if real else None


def score_review(review: CaseReview, review_id: int) -> dict:
    """Score every criterion that was actually checked, then save per-criterion and per-case rows."""
    if review.run_id:
        db.set_trace_url(review_id, observability.run_url(review.run_id))
    notes = _notes_for(review.case_id)
    jobs, keys = [], []
    for r in review.results:
        trace = review.retrieval.get(r.criterion_id)
        if not trace or not trace.reranked or r.reasoning.startswith("Automatic check failed"):
            continue
        # Reasoning legitimately cites both the policy and the patient's notes, so both count as context.
        jobs.append(("faithfulness", {"user_input": r.criterion_text, "response": r.reasoning,
                                      "retrieved_contexts": [h.text for h in trace.reranked] + ([notes] if notes else [])}))
        jobs.append(("context_precision", {"user_input": r.criterion_text, "response": r.reasoning,
                                           "retrieved_contexts": [h.text for h in trace.reranked]}))
        keys += [r.criterion_id, r.criterion_id]
    rows, per_metric = [], {"faithfulness": [], "context_precision": []}
    for (metric, _), cid, value in zip(jobs, keys, score_many(jobs)):
        per_metric[metric].append(value)
        rows.append({"case_id": review.case_id, "review_id": review_id, "criterion_id": cid,
                     "metric": metric, "score": value})
    summary = {m: _avg(v) for m, v in per_metric.items()}
    rows += [{"case_id": review.case_id, "review_id": review_id, "criterion_id": CASE_LEVEL, "metric": m,
              "score": s} for m, s in summary.items()]
    db.save_scores(rows)
    for m, s in summary.items():
        observability.add_feedback(review.run_id, m, s)
    return summary


def score_chat(case_id: str, review_id: int | None, question: str, answer: str, run_id: str | None) -> float | None:
    value = score_many([("answer_relevancy", {"user_input": question, "response": answer})])[0]
    db.save_scores([{"case_id": case_id, "review_id": review_id, "criterion_id": "chat",
                     "metric": "answer_relevancy", "score": value}])
    observability.add_feedback(run_id, "answer_relevancy", value)
    return value


def _notes_for(case_id: str) -> str | None:
    found = db.get_case(case_id)
    return found[0].clinical_notes if found else None
