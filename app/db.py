"""SQLite storage: every chat answer (with its sources), its RAGAS quality scores, and offline eval summaries.

Each function opens its own short connection, so it is safe to call from API threads and background jobs.
"""

import json
import sqlite3
from datetime import datetime, timezone

from app import config
from app.models import ChatAnswer

SCHEMA = """
CREATE TABLE IF NOT EXISTS answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT NOT NULL, verdict TEXT NOT NULL, data TEXT NOT NULL,
    run_id TEXT, trace_url TEXT, total_ms REAL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS eval_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT, answer_id INTEGER NOT NULL, metric TEXT NOT NULL, score REAL,
    timestamp TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS eval_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, summary TEXT NOT NULL, ts TEXT NOT NULL);
"""


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


def save_answer(a: ChatAnswer) -> int:
    with connect() as c:
        cur = c.execute("INSERT INTO answers (question, verdict, data, run_id, total_ms, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (a.question, a.verdict, a.model_dump_json(), a.run_id, a.timings_ms.get("total"), now()))
        return cur.lastrowid


def get_answer(answer_id: int) -> dict | None:
    with connect() as c:
        row = c.execute("SELECT * FROM answers WHERE id=?", (answer_id,)).fetchone()
    return dict(row) if row else None


def set_trace_url(answer_id: int, url: str | None) -> None:
    with connect() as c:
        c.execute("UPDATE answers SET trace_url=? WHERE id=?", (url, answer_id))


def all_answers() -> list[dict]:
    with connect() as c:
        return [dict(r) for r in c.execute(
            "SELECT id, question, verdict, run_id, trace_url, total_ms, created_at FROM answers").fetchall()]


def save_scores(rows: list[dict]) -> None:
    """rows: {answer_id, metric, score (None if scoring failed)}"""
    with connect() as c:
        c.executemany("INSERT INTO eval_scores (answer_id, metric, score, timestamp) "
                      "VALUES (:answer_id, :metric, :score, :timestamp)", [{**r, "timestamp": now()} for r in rows])


def get_scores(answer_id: int) -> list[dict]:
    with connect() as c:
        return [dict(r) for r in c.execute("SELECT * FROM eval_scores WHERE answer_id=?", (answer_id,)).fetchall()]


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
