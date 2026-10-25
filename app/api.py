"""FastAPI backend for the Streamlit UI. Reviews run in the request; RAGAS scoring runs afterwards in a
background task so it never slows the answer.

Run: uv run uvicorn app.api:app --reload
"""

import json
import logging
from contextlib import asynccontextmanager
from statistics import quantiles

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from app import actions, chat, config, db, graph_store, live_eval, observability
from app.guardrails import BANNER, GuardrailError, clean_text, redact_phi
from app.models import CaseRequest, LabeledCase

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    # Load the graph, indexes and local models once, so the first review isn't slow.
    try:
        graph_store.load_graph()
        from app.retriever import get_retriever
        get_retriever()
    except FileNotFoundError as e:
        log.warning("Startup: %s", e)
    if not config.GROQ_API_KEY:
        log.warning("GROQ_API_KEY is not set: reviews will fall back to SEND_TO_CLINICIAN.")
    yield


app = FastAPI(title="PriorAuth Copilot", lifespan=lifespan)


class ChatBody(BaseModel):
    message: str
    history: list[dict] = []


class ActionBody(BaseModel):
    action: str  # confirm | override | finalize
    outcome: str | None = None
    reason: str | None = None
    letter_text: str | None = None
    reviewer: str = "reviewer"


class NoteBody(BaseModel):
    note_line: str


def _case_or_404(case_id: str) -> tuple[CaseRequest, str]:
    found = db.get_case(case_id)
    if not found:
        raise HTTPException(404, f"Case {case_id} not found")
    return found


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "groq_key_set": bool(config.GROQ_API_KEY), "tracing": observability.TRACING_ENABLED,
            "sample_data": config.USE_SAMPLE_DATA, "banner": BANNER}


@app.get("/api/cases")
def list_cases() -> list[dict]:
    return db.list_cases()


@app.post("/api/cases")
def add_case(case: CaseRequest) -> dict:
    # Personal identifiers are removed here, before the note is saved or ever sent to the LLM.
    notes, removed = redact_phi(clean_text(case.clinical_notes))
    case = case.model_copy(update={"clinical_notes": notes})
    db.upsert_case(case)
    db.audit(case.case_id, "reviewer", "case_added", {"cpt_code": case.cpt_code, "removed": removed})
    return {"case_id": case.case_id, "status": db.get_case(case.case_id)[1], "removed": removed}


@app.post("/api/cases/load_samples")
def load_samples() -> dict:
    cases = [LabeledCase.model_validate(c) for c in json.loads(config.CASES_FILE.read_text())]
    for c in cases:
        if not db.get_case(c.case_id):
            db.upsert_case(CaseRequest.model_validate(c.model_dump()))
            db.audit(c.case_id, "system", "sample_loaded")
    return {"loaded": len(cases)}


@app.post("/api/cases/{case_id}/review")
def review(case_id: str, background: BackgroundTasks) -> dict:
    from app.pipeline import review_case
    case, _ = _case_or_404(case_id)
    try:
        result = review_case(case)
    except GuardrailError as e:
        db.audit(case_id, "system", "rejected", {"reason": str(e)})
        raise HTTPException(422, str(e))
    if config.LIVE_EVAL:
        background.add_task(live_eval.score_review, result, result.review_id)
    return {"review_id": result.review_id, "review": result.model_dump(), "banner": BANNER}


@app.get("/api/cases/{case_id}")
def get_case(case_id: str) -> dict:
    case, status = _case_or_404(case_id)
    latest = db.latest_review(case_id)
    return {"case": case.model_dump(), "status": status,
            "review": latest["review"].model_dump() if latest else None,
            "review_id": latest["review_id"] if latest else None,
            "last_action": db.latest_action(case_id), "audit": db.get_audit(case_id), "banner": BANNER}


@app.post("/api/cases/{case_id}/chat")
def chat_route(case_id: str, body: ChatBody, background: BackgroundTasks) -> dict:
    case, _ = _case_or_404(case_id)
    latest = db.latest_review(case_id)
    try:
        message, _ = redact_phi(clean_text(body.message))
        out = chat.answer(case, latest["review"] if latest else None, message, body.history)
    except GuardrailError as e:
        db.audit(case_id, "system", "chat_rejected", {"reason": str(e)})
        raise HTTPException(422, str(e))
    db.audit(case_id, "reviewer", "chat", {"mode": out["mode"], "question": body.message[:200]})
    if out["mode"] == "chat" and config.LIVE_EVAL:
        background.add_task(live_eval.score_chat, case_id, latest["review_id"] if latest else None,
                            body.message, out["answer"], out["run_id"])
    return out


@app.post("/api/cases/{case_id}/add_note")
def add_note(case_id: str, body: NoteBody) -> dict:
    """The reviewer's "Add to case" button after a what-if: append the line to the saved notes."""
    case, _ = _case_or_404(case_id)
    if not body.note_line.strip():
        raise HTTPException(422, "Note is empty.")
    updated = case.model_copy(update={"clinical_notes": f"{case.clinical_notes}\n{body.note_line.strip()}"})
    db.upsert_case(updated)
    db.audit(case_id, "reviewer", "note_added", {"note_line": body.note_line})
    return {"case": updated.model_dump()}


@app.post("/api/cases/{case_id}/action")
def action(case_id: str, body: ActionBody) -> dict:
    _case_or_404(case_id)
    try:
        return actions.handle_action(case_id, body.action, body.outcome, body.reason, body.letter_text, body.reviewer)
    except actions.ActionError as e:
        raise HTTPException(422, str(e))


@app.get("/api/graph")
def graph() -> dict:
    return graph_store.graph_data()


@app.get("/api/cases/{case_id}/scores")
def scores(case_id: str) -> dict:
    """Live RAGAS scores for the latest review. status is 'scoring' until the background job saves rows."""
    _case_or_404(case_id)
    latest = db.latest_review(case_id)
    if not latest:
        return {"status": "no_review", "case": {}, "criteria": {}, "chat": [], "trace_url": None}
    rows = db.get_scores(case_id, latest["review_id"])
    review_rows = [r for r in rows if r["criterion_id"] != "chat"]
    case_level = {r["metric"]: r["score"] for r in review_rows if r["criterion_id"] == live_eval.CASE_LEVEL}
    criteria: dict[str, dict] = {}
    for r in review_rows:
        if r["criterion_id"] != live_eval.CASE_LEVEL:
            criteria.setdefault(r["criterion_id"], {})[r["metric"]] = r["score"]
    return {"status": "done" if case_level else "scoring", "case": case_level, "criteria": criteria,
            "chat": [r["score"] for r in rows if r["criterion_id"] == "chat"],
            "trace_url": latest["trace_url"]}


def _pct(values: list[float], p: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 1)
    return round(quantiles(values, n=100, method="inclusive")[p - 1], 1)


@app.get("/api/metrics")
def metrics() -> dict:
    all_rows = db.all_scores()
    case_rows = [r for r in all_rows if r["criterion_id"] in (live_eval.CASE_LEVEL, "chat")]
    live = {}
    for m in ("faithfulness", "answer_relevancy", "context_precision"):
        vals = [r["score"] for r in case_rows if r["metric"] == m and r["score"] is not None]
        live[m] = round(sum(vals) / len(vals), 3) if vals else None
    trend = [{"timestamp": r["timestamp"], "metric": r["metric"], "score": r["score"]}
             for r in case_rows if r["score"] is not None]

    reviews = db.all_reviews("review")
    url_by_review = {r["review_id"]: r["trace_url"] for r in reviews}
    faith = [r for r in all_rows if r["criterion_id"] == live_eval.CASE_LEVEL and r["metric"] == "faithfulness"
             and r["score"] is not None]
    lowest = [{"case_id": r["case_id"], "review_id": r["review_id"], "faithfulness": r["score"],
               "trace_url": url_by_review.get(r["review_id"])} for r in sorted(faith, key=lambda r: r["score"])[:10]]

    totals = [r["timings_ms"].get("total", 0) for r in reviews]
    stages = sorted({k for r in reviews for k in r["timings_ms"]})
    by_stage = {s: {"p50": _pct([r["timings_ms"][s] for r in reviews if s in r["timings_ms"]], 50),
                    "p95": _pct([r["timings_ms"][s] for r in reviews if s in r["timings_ms"]], 95)} for s in stages}
    outcomes: dict[str, int] = {}
    for r in reviews:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    decided = [a for a in db.all_actions() if a["action"] in ("confirm", "override")]
    overrides = sum(1 for a in decided if a["action"] == "override")
    return {
        "live": live, "trend": trend, "lowest": lowest, "offline": db.last_eval_run(),
        "ops": {"reviews": len(reviews), "latency_p50_ms": _pct(totals, 50), "latency_p95_ms": _pct(totals, 95),
                "by_stage": by_stage, "outcomes": outcomes, "decisions": len(decided),
                "override_rate": round(overrides / len(decided), 3) if decided else None,
                "langsmith_project_url": observability.project_url()},
    }
