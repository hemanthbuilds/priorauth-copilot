"""Background quality scoring with RAGAS after every chat answer. Runs in a FastAPI BackgroundTask, so it never
slows the answer. Scores go to SQLite and, when tracing is on, to the answer's LangSmith trace as feedback.
If scoring fails (e.g. a rate limit), a null score is saved and logged.

Metrics (no answer key needed):
- faithfulness: is the answer backed by its context (the retrieved policy passages + the doctor's note)?
- answer_relevancy: does the answer address the question?
- context_precision: were the retrieved policy passages relevant to the question?
"""

import asyncio
import logging
import threading

from app import config, db, observability
from app.models import ChatAnswer

log = logging.getLogger("live_eval")
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


def score_answer(result: ChatAnswer, answer_id: int, note: str = "") -> dict:
    """Score one chat answer and attach the scores to its LangSmith trace."""
    if result.run_id:
        db.set_trace_url(answer_id, observability.run_url(result.run_id))
    passages = [h.text for h in result.retrieval.reranked]
    jobs = [("answer_relevancy", {"user_input": result.question, "response": result.answer})]
    if passages:
        jobs += [("faithfulness", {"user_input": result.question, "response": result.answer,
                                   "retrieved_contexts": passages + ([note] if note else [])}),
                 ("context_precision", {"user_input": result.question, "response": result.answer,
                                        "retrieved_contexts": passages})]
    scores = {metric: value for (metric, _), value in zip(jobs, score_many(jobs))}
    db.save_scores([{"answer_id": answer_id, "metric": m, "score": v} for m, v in scores.items()])
    for m, v in scores.items():
        observability.add_feedback(result.run_id, m, v)
    return scores
