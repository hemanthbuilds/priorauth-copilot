"""SQLite storage: cases, reviews (with criterion results), reviewer actions, the audit log, and eval scores.

Each function opens its own short connection, so it is safe to call from API threads and background jobs.
"""

import json
import sqlite3
from datetime import datetime, timezone

from app import config
from app.models import CaseRequest, CaseReview

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY, data TEXT NOT NULL, status TEXT NOT NULL,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL, mode TEXT NOT NULL, outcome TEXT NOT NULL,
    data TEXT NOT NULL, run_id TEXT, trace_url TEXT, total_ms REAL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS criterion_results (
    review_id INTEGER NOT NULL, case_id TEXT NOT NULL, criterion_id TEXT NOT NULL, status TEXT NOT NULL,
    confidence TEXT NOT NULL, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL, review_id INTEGER, action TEXT NOT NULL,
    ai_outcome TEXT, final_outcome TEXT, reason TEXT, letter_text TEXT, finalized INTEGER DEFAULT 0,
    actor TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, actor TEXT NOT NULL, event TEXT NOT NULL,
    details TEXT, ts TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS eval_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL, review_id INTEGER, criterion_id TEXT,
    metric TEXT NOT NULL, score REAL, timestamp TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS eval_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, summary TEXT NOT NULL, ts TEXT NOT NULL);
"""

# Case statuses shown in the queue, and which one each final outcome leads to.
STATUS_FOR_OUTCOME = {"APPROVE": "Approved", "REQUEST_INFO": "Info requested", "SEND_TO_CLINICIAN": "With clinician"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with connect() as c:
        c.executescript(SCHEMA)


# ---------- cases ----------

def upsert_case(case: CaseRequest, status: str = "New") -> None:
    with connect() as c:
        c.execute("""INSERT INTO cases (case_id, data, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)
                     ON CONFLICT(case_id) DO UPDATE SET data=excluded.data, updated_at=excluded.updated_at""",
                  (case.case_id, case.model_dump_json(), status, now(), now()))


def get_case(case_id: str) -> tuple[CaseRequest, str] | None:
    with connect() as c:
        row = c.execute("SELECT data, status FROM cases WHERE case_id=?", (case_id,)).fetchone()
    return (CaseRequest.model_validate_json(row["data"]), row["status"]) if row else None


def list_cases() -> list[dict]:
    with connect() as c:
        rows = c.execute("SELECT data, status, updated_at FROM cases ORDER BY case_id").fetchall()
    out = []
    for r in rows:
        case = json.loads(r["data"])
        out.append({"case_id": case["case_id"], "cpt_code": case["cpt_code"], "patient_name": case["patient_name"],
                    "status": r["status"], "updated_at": r["updated_at"]})
    return out


def set_status(case_id: str, status: str) -> None:
    with connect() as c:
        c.execute("UPDATE cases SET status=?, updated_at=? WHERE case_id=?", (status, now(), case_id))


# ---------- reviews ----------

def save_review(review: CaseReview) -> int:
    with connect() as c:
        cur = c.execute("""INSERT INTO reviews (case_id, mode, outcome, data, run_id, total_ms, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (review.case_id, review.mode, review.decision.outcome, review.model_dump_json(),
                         review.run_id, review.timings_ms.get("total"), now()))
        review_id = cur.lastrowid
        c.executemany("INSERT INTO criterion_results VALUES (?, ?, ?, ?, ?, ?)",
                      [(review_id, review.case_id, r.criterion_id, r.status, r.confidence, r.model_dump_json())
                       for r in review.results])
    return review_id


def latest_review(case_id: str) -> dict | None:
    """Most recent normal review (not what-if) with its id and trace URL."""
    with connect() as c:
        row = c.execute("SELECT * FROM reviews WHERE case_id=? AND mode='review' ORDER BY id DESC LIMIT 1",
                        (case_id,)).fetchone()
    if not row:
        return None
    return {"review_id": row["id"], "review": CaseReview.model_validate_json(row["data"]),
            "run_id": row["run_id"], "trace_url": row["trace_url"], "created_at": row["created_at"]}


def set_trace_url(review_id: int, url: str | None) -> None:
    with connect() as c:
        c.execute("UPDATE reviews SET trace_url=? WHERE id=?", (url, review_id))


def all_reviews(mode: str = "review") -> list[dict]:
    with connect() as c:
        rows = c.execute("SELECT id, case_id, outcome, data, trace_url, created_at FROM reviews WHERE mode=?",
                         (mode,)).fetchall()
    return [{"review_id": r["id"], "case_id": r["case_id"], "outcome": r["outcome"],
             "timings_ms": json.loads(r["data"])["timings_ms"], "trace_url": r["trace_url"],
             "created_at": r["created_at"]} for r in rows]


# ---------- actions + audit ----------

def save_action(case_id: str, review_id: int | None, action: str, ai_outcome: str | None, final_outcome: str | None,
                reason: str | None, letter_text: str | None, actor: str, finalized: bool = False) -> int:
    with connect() as c:
        cur = c.execute("""INSERT INTO actions (case_id, review_id, action, ai_outcome, final_outcome, reason,
                           letter_text, finalized, actor, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (case_id, review_id, action, ai_outcome, final_outcome, reason, letter_text,
                         int(finalized), actor, now()))
        return cur.lastrowid


def latest_action(case_id: str) -> dict | None:
    with connect() as c:
        row = c.execute("SELECT * FROM actions WHERE case_id=? ORDER BY id DESC LIMIT 1", (case_id,)).fetchone()
    return dict(row) if row else None


def all_actions() -> list[dict]:
    with connect() as c:
        return [dict(r) for r in c.execute("SELECT * FROM actions").fetchall()]


def audit(case_id: str | None, actor: str, event: str, details: dict | None = None) -> None:
    with connect() as c:
        c.execute("INSERT INTO audit_log (case_id, actor, event, details, ts) VALUES (?, ?, ?, ?, ?)",
                  (case_id, actor, event, json.dumps(details or {}), now()))


def get_audit(case_id: str) -> list[dict]:
    with connect() as c:
        rows = c.execute("SELECT actor, event, details, ts FROM audit_log WHERE case_id=? ORDER BY id",
                         (case_id,)).fetchall()
    return [{**dict(r), "details": json.loads(r["details"] or "{}")} for r in rows]


# ---------- eval scores ----------

def save_scores(rows: list[dict]) -> None:
    """rows: {case_id, review_id, criterion_id, metric, score (None if scoring failed)}"""
    with connect() as c:
        c.executemany("""INSERT INTO eval_scores (case_id, review_id, criterion_id, metric, score, timestamp)
                         VALUES (:case_id, :review_id, :criterion_id, :metric, :score, :timestamp)""",
                      [{**r, "timestamp": now()} for r in rows])


def get_scores(case_id: str, review_id: int | None = None) -> list[dict]:
    q, args = "SELECT * FROM eval_scores WHERE case_id=?", [case_id]
    if review_id is not None:
        q, args = q + " AND review_id=?", args + [review_id]
    with connect() as c:
        return [dict(r) for r in c.execute(q, args).fetchall()]


def all_scores() -> list[dict]:
    with connect() as c:
        return [dict(r) for r in c.execute("SELECT * FROM eval_scores ORDER BY id").fetchall()]


def save_eval_run(summary: dict) -> None:
    with connect() as c:
        c.execute("INSERT INTO eval_runs (summary, ts) VALUES (?, ?)", (json.dumps(summary), now()))


def last_eval_run() -> dict | None:
    with connect() as c:
        row = c.execute("SELECT summary, ts FROM eval_runs ORDER BY id DESC LIMIT 1").fetchone()
    return {"ts": row["ts"], **json.loads(row["summary"])} if row else None
