"""The evaluation script's bookkeeping, run on the fixture cases with the fake LLM/retriever.
These numbers come from fakes and only test the code paths; they are NOT real model results."""

from app import live_eval
from eval import run_eval
from tests.conftest import CASES


def test_evaluate_and_report(fake_llm, fake_retriever, monkeypatch, tmp_path):
    monkeypatch.setattr("app.config.DB_PATH", tmp_path / "eval.db")
    monkeypatch.setattr(live_eval, "score_many", lambda jobs: [0.5] * len(jobs))
    from app import db
    db.init_db()
    summary = run_eval.evaluate(CASES, ["full", "vector_only"], delay=0)
    assert summary["full"]["verdict_accuracy"] == 1.0  # the fake LLM echoes the labels
    assert summary["full"]["faithfulness"] == 0.5 and summary["full"]["unverified_quote_rate"] == 0
    assert len(summary["cases"]) == len(CASES) and [a["mode"] for a in summary["ablation"]] == ["full", "vector_only"]
    md = run_eval.report_md(summary)
    assert "Verdict accuracy" in md and "## Ablation" in md and "Case by case" in md
    assert db.last_eval_run()["n_cases"] == len(CASES)


def test_ai_service_failures_are_excluded_not_counted_wrong(fake_llm, fake_retriever, monkeypatch, tmp_path):
    monkeypatch.setattr("app.config.DB_PATH", tmp_path / "eval.db")
    from app import db
    db.init_db()
    fake_llm.fail = True
    summary = run_eval.evaluate(CASES[:2], ["full"], delay=0, ragas=False)
    assert len(summary["errored"]) == 2 and summary["full"]["verdict_accuracy"] is None
