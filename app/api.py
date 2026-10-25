"""FastAPI backend for the chat UI. The answer is computed in the request; RAGAS scoring runs afterwards in a
background task so it never slows the answer.

Run: uv run uvicorn app.api:app --port 8010
"""

import logging
from contextlib import asynccontextmanager
from statistics import quantiles

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from app import chat, config, db, live_eval, observability, policy_store
from app.guardrails import BANNER, GuardrailError

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    # Load the policy table, search index and local models once, so the first question isn't slow.
    try:
        policy_store.load()
        from app.retriever import get_retriever
        get_retriever()
    except FileNotFoundError as e:
        log.warning("Startup: %s", e)
    if not config.GROQ_API_KEY:
        log.warning("GROQ_API_KEY is not set: the chatbot can't answer until it is.")
    yield


app = FastAPI(title="PriorAuth Copilot", lifespan=lifespan)


class ChatBody(BaseModel):
    question: str
    note: str = ""
    cpt: str | None = None
    icd_codes: list[str] = []
    history: list[dict] = []


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "groq_key_set": bool(config.GROQ_API_KEY), "tracing": observability.TRACING_ENABLED,
            "banner": BANNER}


@app.get("/api/procedures")
def procedures() -> dict:
    """{cpt: procedure name} for every procedure with a policy."""
    return policy_store.procedures()


@app.post("/api/chat")
def chat_route(body: ChatBody, background: BackgroundTasks) -> dict:
    try:
        result = chat.answer(body.question, body.note, body.cpt, body.icd_codes or None, body.history)
    except GuardrailError as e:
        raise HTTPException(422, str(e))
    result.answer_id = db.save_answer(result)
    if config.LIVE_EVAL and result.verdict and result.retrieval.reranked:
        background.add_task(live_eval.score_answer, result, result.answer_id, body.note)
    return result.model_dump()


@app.get("/api/answers/{answer_id}/scores")
def scores(answer_id: int) -> dict:
    """Live RAGAS scores for one answer. status is 'scoring' until the background job saves them."""
    row = db.get_answer(answer_id)
    if not row:
        raise HTTPException(404, f"Answer {answer_id} not found")
    rows = db.get_scores(answer_id)
    return {"status": "done" if rows else "scoring", "scores": {r["metric"]: r["score"] for r in rows},
            "trace_url": row["trace_url"]}


def _pct(values: list[float], p: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 1)
    return round(quantiles(values, n=100, method="inclusive")[p - 1], 1)


@app.get("/api/metrics")
def metrics() -> dict:
    scores = [r for r in db.all_scores() if r["score"] is not None]
    live = {m: (round(sum(v) / len(v), 3) if (v := [r["score"] for r in scores if r["metric"] == m]) else None)
            for m in ("faithfulness", "answer_relevancy", "context_precision")}
    answers = db.all_answers()
    totals = [a["total_ms"] for a in answers if a["total_ms"]]
    verdicts: dict[str, int] = {}
    for a in answers:
        verdicts[a["verdict"]] = verdicts.get(a["verdict"], 0) + 1
    return {"live": live, "trend": [{"timestamp": r["timestamp"], "metric": r["metric"], "score": r["score"]}
                                    for r in scores],
            "offline": db.last_eval_run(),
            "ops": {"answers": len(answers), "latency_p50_ms": _pct(totals, 50), "latency_p95_ms": _pct(totals, 95),
                    "verdicts": verdicts, "langsmith_project_url": observability.project_url()}}
