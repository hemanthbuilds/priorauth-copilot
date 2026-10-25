"""The eval scripts' bookkeeping, run on the sample cases with the fake LLM/retriever.
These numbers are from fakes and only test the code paths; they are NOT real model results."""

from app import live_eval
from eval import run_eval
from eval.make_cases import rule_outcome
from tests.conftest import SAMPLE_CASES


def test_sample_labels_follow_the_decision_rules():
    for case in SAMPLE_CASES:
        assert rule_outcome(case) == case.expected_outcome, case.case_id


def test_evaluate_and_report(fake_llm, fake_retriever, monkeypatch, tmp_path):
    monkeypatch.setattr("app.config.DB_PATH", tmp_path / "eval.db")
    monkeypatch.setattr(live_eval, "score_many", lambda jobs: [0.5] * len(jobs))
    from app import db
    db.init_db()
    summary = run_eval.evaluate(SAMPLE_CASES, ["full", "vector_only"], delay=0)
    full = summary["full"]
    assert full["decision_accuracy"] == 1.0 and full["criterion_accuracy"] == 1.0  # fake LLM echoes the labels
    assert full["faithfulness"] == 0.5 and full["context_recall"] == 0.5
    assert sum(sum(row.values()) for row in summary["confusion"].values()) == len(SAMPLE_CASES)
    assert [a["mode"] for a in summary["ablation"]] == ["full", "vector_only"]
    md = run_eval.report_md(summary)
    assert "Decision accuracy" in md and "## Ablation" in md and "Latency per stage" in md
    assert db.last_eval_run()["n_cases"] == len(SAMPLE_CASES)


def test_ai_service_failures_are_excluded_not_counted_wrong(fake_llm, fake_retriever, monkeypatch, tmp_path):
    monkeypatch.setattr("app.config.DB_PATH", tmp_path / "eval.db")
    from app import db
    db.init_db()
    fake_llm.fail = True  # every LLM call fails, like a rate limit or network drop
    summary = run_eval.evaluate(SAMPLE_CASES[:2], ["full"], delay=0, ragas=False)
    assert len(summary["errored"]) == 2 and summary["full"]["decision_accuracy"] is None
    assert "Errored cases" in run_eval.report_md(summary)
