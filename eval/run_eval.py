"""Offline evaluation on the human-labeled cases in data/cases/cases.json.

Reports decision accuracy (+ confusion matrix), criterion accuracy, RAGAS faithfulness and context
recall, a retrieval check (does the reranked top 3 contain the criterion's source page?), p50/p95
latency per stage, and an ablation:
  full         - hybrid search in the procedure's policy (BM25 + vector + RRF + rerank)
  all_policies - same hybrid search but over ALL policies (no filter to the procedure's policy)
  vector_only  - the procedure's policy, vector top 3 only (no BM25 / RRF / rerank)
Writes results/eval_report.md and saves the summary to SQLite for the Metrics page.
With LangSmith keys set, the cases are uploaded as a dataset and each mode runs as an experiment.

Run: uv run python -m eval.run_eval [--modes full,all_policies,vector_only] [--limit N] [--delay 2] [--no-ragas]
"""

import argparse
import json
import sys
import time
from statistics import quantiles

from app import config, db, live_eval, observability
from app.models import CaseRequest, CaseReview, LabeledCase
from app.pipeline import review_case

DATASET = f"priorauth-copilot-{'sample' if config.USE_SAMPLE_DATA else 'cms'}-cases"
OUTCOMES = ["APPROVE", "REQUEST_INFO", "SEND_TO_CLINICIAN"]


def load_cases(limit: int | None = None) -> list[LabeledCase]:
    cases = [LabeledCase.model_validate(c) for c in json.loads(config.CASES_FILE.read_text())]
    return cases[:limit] if limit else cases


def run_one(case: LabeledCase, mode: str, delay: float, tries: int = 3) -> CaseReview:
    """Run one review; retry on unexpected errors (e.g. Groq rate limits beyond the SDK's own retries)."""
    request = CaseRequest.model_validate(case.model_dump())
    for attempt in range(tries):
        try:
            review = review_case(request, mode="eval", retrieval_mode=mode, save=False)
            time.sleep(delay)
            return review
        except Exception as e:
            if attempt == tries - 1:
                raise
            print(f"  retrying {case.case_id} after error: {e}")
            time.sleep(delay * 5)


def run_mode(mode: str, cases: list[LabeledCase], delay: float) -> list[tuple[LabeledCase, CaseReview]]:
    if observability.TRACING_ENABLED:
        return _run_mode_langsmith(mode, cases, delay)
    pairs = []
    for c in cases:
        print(f"  [{mode}] {c.case_id}")
        pairs.append((c, run_one(c, mode, delay)))
    return pairs


def _run_mode_langsmith(mode: str, cases: list[LabeledCase], delay: float):
    """Same as run_mode, but as a LangSmith experiment so modes can be compared side by side there."""
    client = observability.get_client()
    if not client.has_dataset(dataset_name=DATASET):
        ds = client.create_dataset(DATASET, description="Synthetic prior auth cases with human labels")
        client.create_examples(dataset_id=ds.id, examples=[
            {"inputs": {"case_id": c.case_id}, "outputs": {"expected_outcome": c.expected_outcome,
                                                           "expected_statuses": c.expected_statuses}} for c in cases])
    by_id = {c.case_id: c for c in cases}
    reviews: dict[str, CaseReview] = {}

    def target(inputs: dict) -> dict:
        case = by_id.get(inputs["case_id"])
        if case is None:
            return {"outcome": None, "statuses": {}}
        print(f"  [{mode}] {case.case_id}")
        reviews[case.case_id] = run_one(case, mode, delay)
        return {"outcome": reviews[case.case_id].decision.outcome,
                "statuses": {r.criterion_id: r.status for r in reviews[case.case_id].results}}

    def decision_correct(outputs: dict, reference_outputs: dict) -> dict:
        return {"key": "decision_correct", "score": int(outputs["outcome"] == reference_outputs["expected_outcome"])}

    def criterion_accuracy(outputs: dict, reference_outputs: dict) -> dict:
        exp = reference_outputs["expected_statuses"]
        hits = sum(outputs["statuses"].get(k) == v for k, v in exp.items())
        return {"key": "criterion_accuracy", "score": hits / len(exp) if exp else None}

    client.evaluate(target, data=DATASET, evaluators=[decision_correct, criterion_accuracy],
                    experiment_prefix=f"priorauth-{mode}", max_concurrency=1, metadata={"retrieval_mode": mode})
    return [(by_id[cid], r) for cid, r in reviews.items()]


def pct(values: list[float], p: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 1)
    return round(quantiles(values, n=100, method="inclusive")[p - 1], 1)


def errored(review: CaseReview) -> bool:
    """The AI service failed (rate limit, network) for this case: that says nothing about the app's accuracy."""
    return any(r.startswith("Automatic check failed") for r in review.decision.reasons)


def summarize(pairs: list[tuple[LabeledCase, CaseReview]]) -> dict:
    """Accuracy, retrieval check, off-policy leakage and latency for one mode (no LLM calls)."""
    from app import policy_store
    n = len(pairs)
    decision_hits = sum(r.decision.outcome == c.expected_outcome for c, r in pairs)
    crit_total = crit_hits = retr_total = retr_hits = chunks = off_policy = 0
    for c, r in pairs:
        got = {x.criterion_id: x.status for x in r.results}
        for cid, status in c.expected_statuses.items():
            crit_total += 1
            crit_hits += got.get(cid) == status
        own_files = set(policy_store.get_policy(c.cpt_code).source.split(", "))
        pages = {crit.id: crit.page for g in policy_store.get_criteria(c.cpt_code) for crit in g.criteria}
        for cid, tr in r.retrieval.items():
            retr_total += 1
            retr_hits += any(h.page == pages.get(cid) and h.source in own_files for h in tr.reranked)
            chunks += len(tr.reranked)
            off_policy += sum(h.source not in own_files for h in tr.reranked)
    stages = sorted({k for _, r in pairs for k in r.timings_ms})
    return {
        "n": n,
        "decision_accuracy": round(decision_hits / n, 3) if n else None,
        "criterion_accuracy": round(crit_hits / crit_total, 3) if crit_total else None,
        "retrieval_hit_rate": round(retr_hits / retr_total, 3) if retr_total else None,
        "off_policy_chunk_rate": round(off_policy / chunks, 3) if chunks else None,
        "latency": {s: {"p50": pct([r.timings_ms[s] for _, r in pairs if s in r.timings_ms], 50),
                        "p95": pct([r.timings_ms[s] for _, r in pairs if s in r.timings_ms], 95)} for s in stages},
    }


def confusion(pairs) -> dict:
    m = {e: {p: 0 for p in OUTCOMES} for e in OUTCOMES}
    for c, r in pairs:
        m[c.expected_outcome][r.decision.outcome] += 1
    return m


def ragas_scores(pairs) -> dict:
    """Faithfulness (reasoning vs policy context + notes) and context recall (reference = the criterion's
    policy wording from criteria.json) over every criterion that was actually checked."""
    jobs = []
    for c, r in pairs:
        for x in r.results:
            tr = r.retrieval.get(x.criterion_id)
            if not tr or not tr.reranked or x.reasoning.startswith("Automatic check failed"):
                continue
            jobs.append(("faithfulness", {"user_input": x.criterion_text, "response": x.reasoning,
                                          "retrieved_contexts": [h.text for h in tr.reranked] + [c.clinical_notes]}))
            jobs.append(("context_recall", {"user_input": x.criterion_text, "reference": x.criterion_text,
                                            "retrieved_contexts": [h.text for h in tr.reranked]}))
    scores = live_eval.score_many(jobs)
    out = {}
    for metric in ("faithfulness", "context_recall"):
        vals = [s for (m, _), s in zip(jobs, scores) if m == metric and s is not None]
        out[metric] = round(sum(vals) / len(vals), 3) if vals else None
        out[f"{metric}_scored"] = f"{len(vals)}/{sum(1 for m, _ in jobs if m == metric)}"
    return out


def report_md(summary: dict) -> str:
    f = summary["full"]

    def v(x, pct_=False):
        return "n/a" if x is None else (f"{x:.0%}" if pct_ else f"{x}")

    lines = [
        "# PriorAuth Copilot: offline eval report", "",
        f"- Run: {summary['ts']}", f"- Cases: {summary['n_cases']} ({'fictional sample policies' if summary['sample_data'] else 'CMS policies'})",
        f"- Model: {summary['model']} (RAGAS judge: {summary.get('ragas_model', 'n/a')})",
        f"- Errored cases (AI service failed; excluded from all numbers): "
        f"{', '.join(summary.get('errored', [])) or 'none'}", "",
        "## Full pipeline", "",
        "| Metric | Value |", "|---|---|",
        f"| Decision accuracy | {v(f['decision_accuracy'], True)} |",
        f"| Criterion accuracy | {v(f['criterion_accuracy'], True)} |",
        f"| Faithfulness (RAGAS) | {v(f.get('faithfulness'))} (scored {f.get('faithfulness_scored', 'n/a')}) |",
        f"| Context recall (RAGAS) | {v(f.get('context_recall'))} (scored {f.get('context_recall_scored', 'n/a')}) |",
        f"| Retrieval check (source page in reranked top 3) | {v(f['retrieval_hit_rate'], True)} |", "",
        "## Confusion matrix (rows = expected, columns = predicted)", "",
        "| expected \\ predicted | " + " | ".join(OUTCOMES) + " |", "|---|" + "---|" * len(OUTCOMES),
    ]
    for e in OUTCOMES:
        lines.append(f"| {e} | " + " | ".join(str(summary["confusion"][e][p]) for p in OUTCOMES) + " |")
    lines += ["", "## Case by case (full pipeline)", "", "| Case | Expected | Got | Match | Rules matching labels |",
              "|---|---|---|---|---|"]
    for row in summary.get("cases", []):
        lines.append(f"| {row['case_id']} | {row['expected']} | {row['got']} | {'✅' if row['match'] else '❌'} | "
                     f"{row['rules']} |")
    lines += ["", "## Ablation", "",
              "| Mode | Decision acc. | Criterion acc. | Retrieval check | Off-policy chunks | p50 total ms |",
              "|---|---|---|---|---|---|"]
    for a in summary["ablation"]:
        lines.append(f"| {a['mode']} | {v(a['decision_accuracy'], True)} | {v(a['criterion_accuracy'], True)} | "
                     f"{v(a['retrieval_hit_rate'], True)} | {v(a['off_policy_chunk_rate'], True)} | {v(a['p50_ms'])} |")
    lines += ["", "## Latency per stage (full pipeline, ms)", "", "| Stage | p50 | p95 |", "|---|---|---|"]
    for s, t in f["latency"].items():
        lines.append(f"| {s} | {t['p50']} | {t['p95']} |")
    lines += ["", "Notes: `retrieve` and `llm_check` are summed over criteria (they run in parallel, so "
              "`criteria_wall` is the real elapsed time). Faithfulness counts the clinical notes as context, "
              "because the reasoning legitimately cites them."]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="full,all_policies,vector_only")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--delay", type=float, default=2.0, help="seconds between cases (Groq rate limits)")
    ap.add_argument("--no-ragas", action="store_true")
    ap.add_argument("--ragas-cases", type=int, default=None,
                    help="score RAGAS on only the first N cases (saves LLM tokens on free tiers)")
    args = ap.parse_args()
    if not config.GROQ_API_KEY:
        sys.exit("GROQ_API_KEY is not set: the eval needs the real LLM, so it was not run (no results were faked).")

    db.init_db()
    cases = load_cases(args.limit)
    modes = [m.strip() for m in args.modes.split(",")]
    if "full" not in modes:
        modes.insert(0, "full")
    summary = evaluate(cases, modes, args.delay, ragas=not args.no_ragas, ragas_cases=args.ragas_cases)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "eval_report.md").write_text(report_md(summary))
    print(report_md(summary))
    print(f"Wrote {config.RESULTS_DIR / 'eval_report.md'}")


def evaluate(cases: list[LabeledCase], modes: list[str], delay: float, ragas: bool = True,
             ragas_cases: int | None = None) -> dict:
    results, failed = {}, set()
    for mode in modes:
        print(f"Running mode: {mode}")
        pairs = run_mode(mode, cases, delay)
        failed |= {f"{c.case_id} ({mode})" for c, r in pairs if errored(r)}
        results[mode] = [(c, r) for c, r in pairs if not errored(r)]
    full = summarize(results["full"])
    if ragas:
        print("Scoring RAGAS faithfulness + context recall ...")
        full.update(ragas_scores(results["full"][:ragas_cases]))
    else:
        full.update({"faithfulness": None, "context_recall": None})
    ablation = []
    for mode in modes:
        s = full if mode == "full" else summarize(results[mode])
        ablation.append({"mode": mode, "decision_accuracy": s["decision_accuracy"],
                         "criterion_accuracy": s["criterion_accuracy"], "retrieval_hit_rate": s["retrieval_hit_rate"],
                         "off_policy_chunk_rate": s["off_policy_chunk_rate"],
                         "p50_ms": s["latency"].get("total", {}).get("p50")})
    per_case = []
    for c, r in results["full"]:
        got = {x.criterion_id: x.status for x in r.results}
        hits = sum(got.get(k) == v for k, v in c.expected_statuses.items())
        per_case.append({"case_id": c.case_id, "expected": c.expected_outcome, "got": r.decision.outcome,
                         "match": r.decision.outcome == c.expected_outcome, "rules": f"{hits}/{len(c.expected_statuses)}"})
    summary = {"ts": db.now(), "n_cases": len(cases), "model": config.GROQ_MODEL, "ragas_model": config.RAGAS_MODEL,
               "sample_data": config.USE_SAMPLE_DATA, "errored": sorted(failed), "cases": per_case,
               "full": full, "confusion": confusion(results["full"]), "ablation": ablation}
    db.save_eval_run(summary)
    return summary


if __name__ == "__main__":
    main()
