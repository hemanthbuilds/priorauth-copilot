"""Offline evaluation on the labeled cases in data/test_cases.json.

For each case the chatbot is asked the same question a billing person would ask ("Does this note meet the
policy for <procedure>, diagnosis <codes>?") and its verdict is compared with the expected one.

Reports: verdict accuracy + confusion matrix, a case-by-case table, how often an answer contained a quote that
wasn't really in the sources, RAGAS faithfulness and answer relevancy, latency, and an ablation:
  full         - hybrid search in the procedure's policy (BM25 + vector + RRF + rerank)
  all_policies - same hybrid search but over every policy
  vector_only  - the procedure's policy, vector search only (no BM25 / RRF / rerank)
Cases where the AI service failed (rate limit, network) are reported as errored, not counted as wrong.
Writes results/eval_report.md and saves the summary for the Metrics page. With LangSmith keys set, the cases
are uploaded as a dataset and each setup runs as an experiment.

Run: uv run python -m eval.run_eval [--modes full,all_policies,vector_only] [--limit N] [--delay 3] [--ragas-cases N]
"""

import argparse
import json
import sys
import time
from statistics import quantiles

from app import chat, config, db, live_eval, observability, policy_store
from app.models import ChatAnswer, TestCase

DATASET = "priorauth-copilot-chat-cases"
VERDICTS = ["meets", "missing_info", "does_not_meet"]


def load_cases(limit: int | None = None) -> list[TestCase]:
    cases = [TestCase.model_validate(c) for c in json.loads(config.CASES_FILE.read_text())]
    return cases[:limit] if limit else cases


def question_for(case: TestCase) -> str:
    name = policy_store.procedures().get(case.cpt_code, f"CPT {case.cpt_code}")
    return (f"Does this note meet the Medicare policy for {name} (CPT {case.cpt_code}) with diagnosis "
            f"{', '.join(case.icd_codes)}? Check every requirement.")


def run_one(case: TestCase, mode: str, delay: float) -> ChatAnswer:
    result = chat.answer(question_for(case), case.clinical_notes, case.cpt_code, case.icd_codes,
                         retrieval_mode=mode, mode="eval")
    time.sleep(delay)
    return result


def errored(result: ChatAnswer) -> bool:
    return result.answer.startswith("Sorry, I couldn't reach the AI service")


def run_mode(mode: str, cases: list[TestCase], delay: float) -> list[tuple[TestCase, ChatAnswer]]:
    if not observability.TRACING_ENABLED:
        out = []
        for c in cases:
            print(f"  [{mode}] {c.case_id}")
            out.append((c, run_one(c, mode, delay)))
        return out
    # Same thing as a LangSmith experiment, so the setups can be compared side by side there.
    client = observability.get_client()
    if not client.has_dataset(dataset_name=DATASET):
        ds = client.create_dataset(DATASET, description="Synthetic doctor's notes with the expected verdict")
        client.create_examples(dataset_id=ds.id, examples=[
            {"inputs": {"case_id": c.case_id}, "outputs": {"expected_verdict": c.expected_verdict}} for c in cases])
    by_id, results = {c.case_id: c for c in cases}, {}

    def target(inputs: dict) -> dict:
        case = by_id.get(inputs["case_id"])
        if case is None:
            return {"verdict": None}
        print(f"  [{mode}] {case.case_id}")
        results[case.case_id] = run_one(case, mode, delay)
        return {"verdict": results[case.case_id].verdict}

    def verdict_correct(outputs: dict, reference_outputs: dict) -> dict:
        return {"key": "verdict_correct", "score": int(outputs["verdict"] == reference_outputs["expected_verdict"])}

    client.evaluate(target, data=DATASET, evaluators=[verdict_correct], experiment_prefix=f"priorauth-{mode}",
                    max_concurrency=1, metadata={"retrieval_mode": mode})
    return [(by_id[cid], r) for cid, r in results.items()]


def pct(values: list[float], p: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 1)
    return round(quantiles(values, n=100, method="inclusive")[p - 1], 1)


def summarize(pairs: list[tuple[TestCase, ChatAnswer]]) -> dict:
    n = len(pairs)
    hits = sum(r.verdict == c.expected_verdict for c, r in pairs)
    off_policy = sum(1 for c, r in pairs for h in r.retrieval.reranked
                     if h.source not in policy_store.get_policy(c.cpt_code).source.split(", "))
    passages = sum(len(r.retrieval.reranked) for _, r in pairs)
    stages = sorted({k for _, r in pairs for k in r.timings_ms})
    return {
        "n": n,
        "verdict_accuracy": round(hits / n, 3) if n else None,
        "unverified_quote_rate": round(sum(bool(r.unverified_quotes) for _, r in pairs) / n, 3) if n else None,
        "off_policy_passage_rate": round(off_policy / passages, 3) if passages else None,
        "latency": {s: {"p50": pct([r.timings_ms[s] for _, r in pairs if s in r.timings_ms], 50),
                        "p95": pct([r.timings_ms[s] for _, r in pairs if s in r.timings_ms], 95)} for s in stages},
    }


def ragas_scores(pairs) -> dict:
    jobs = []
    for c, r in pairs:
        jobs.append(("answer_relevancy", {"user_input": r.question, "response": r.answer}))
        if r.retrieval.reranked:
            jobs.append(("faithfulness", {"user_input": r.question, "response": r.answer,
                                          "retrieved_contexts": [h.text for h in r.retrieval.reranked] + [c.clinical_notes]}))
    scores = live_eval.score_many(jobs)
    out = {}
    for metric in ("faithfulness", "answer_relevancy"):
        vals = [s for (m, _), s in zip(jobs, scores) if m == metric and s is not None]
        out[metric] = round(sum(vals) / len(vals), 3) if vals else None
        out[f"{metric}_scored"] = f"{len(vals)}/{sum(1 for m, _ in jobs if m == metric)}"
    return out


def evaluate(cases: list[TestCase], modes: list[str], delay: float, ragas: bool = True,
             ragas_cases: int | None = None) -> dict:
    results, failed = {}, set()
    for mode in modes:
        print(f"Running mode: {mode}")
        pairs = run_mode(mode, cases, delay)
        failed |= {f"{c.case_id} ({mode})" for c, r in pairs if errored(r)}
        results[mode] = [(c, r) for c, r in pairs if not errored(r)]
    full = summarize(results["full"])
    full.update(ragas_scores(results["full"][:ragas_cases]) if ragas else {"faithfulness": None, "answer_relevancy": None})
    ablation = []
    for m in modes:
        s = full if m == "full" else summarize(results[m])
        ablation.append({"mode": m, "verdict_accuracy": s["verdict_accuracy"],
                         "unverified_quote_rate": s["unverified_quote_rate"],
                         "off_policy_passage_rate": s["off_policy_passage_rate"],
                         "p50_ms": s["latency"].get("total", {}).get("p50")})
    confusion = {e: {p: 0 for p in VERDICTS + ["not_applicable"]} for e in VERDICTS}
    for c, r in results["full"]:
        confusion[c.expected_verdict][r.verdict] += 1
    per_case = [{"case_id": c.case_id, "expected": c.expected_verdict, "got": r.verdict,
                 "match": r.verdict == c.expected_verdict, "unverified_quotes": len(r.unverified_quotes)}
                for c, r in results["full"]]
    summary = {"ts": db.now(), "n_cases": len(cases), "model": config.GROQ_MODEL, "ragas_model": config.RAGAS_MODEL,
               "errored": sorted(failed), "cases": per_case, "full": full, "confusion": confusion, "ablation": ablation}
    db.save_eval_run(summary)
    return summary


def report_md(summary: dict) -> str:
    f = summary["full"]

    def v(x, as_pct=False):
        return "n/a" if x is None else (f"{x:.0%}" if as_pct else f"{x}")

    lines = ["# PriorAuth Copilot: evaluation report", "",
             f"- Run: {summary['ts']}", f"- Cases: {summary['n_cases']}",
             f"- Model: {summary['model']} (RAGAS judge: {summary['ragas_model']})",
             f"- Errored (AI service failed; excluded): {', '.join(summary['errored']) or 'none'}", "",
             "## Full pipeline", "", "| Metric | Value |", "|---|---|",
             f"| Verdict accuracy | {v(f['verdict_accuracy'], True)} |",
             f"| Answers with an unverified quote | {v(f['unverified_quote_rate'], True)} |",
             f"| Passages from the wrong policy | {v(f['off_policy_passage_rate'], True)} |",
             f"| Faithfulness (RAGAS) | {v(f.get('faithfulness'))} (scored {f.get('faithfulness_scored', 'n/a')}) |",
             f"| Answer relevancy (RAGAS) | {v(f.get('answer_relevancy'))} (scored {f.get('answer_relevancy_scored', 'n/a')}) |",
             "", "## Case by case", "", "| Case | Expected | Got | Match | Unverified quotes |", "|---|---|---|---|---|"]
    lines += [f"| {r['case_id']} | {r['expected']} | {r['got']} | {'✅' if r['match'] else '❌'} | {r['unverified_quotes']} |"
              for r in summary["cases"]]
    cols = VERDICTS + ["not_applicable"]
    lines += ["", "## Confusion matrix (rows = expected, columns = answered)", "",
              "| expected \\ answered | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    lines += [f"| {e} | " + " | ".join(str(summary["confusion"][e][p]) for p in cols) + " |" for e in VERDICTS]
    lines += ["", "## Ablation", "", "| Mode | Verdict acc. | Unverified quotes | Wrong-policy passages | p50 ms |",
              "|---|---|---|---|---|"]
    lines += [f"| {a['mode']} | {v(a['verdict_accuracy'], True)} | {v(a['unverified_quote_rate'], True)} | "
              f"{v(a['off_policy_passage_rate'], True)} | {v(a['p50_ms'])} |" for a in summary["ablation"]]
    lines += ["", "## Latency (full pipeline, ms)", "", "| Stage | p50 | p95 |", "|---|---|---|"]
    lines += [f"| {s} | {t['p50']} | {t['p95']} |" for s, t in f["latency"].items()]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="full,all_policies,vector_only")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--delay", type=float, default=3.0, help="seconds between cases (Groq rate limits)")
    ap.add_argument("--no-ragas", action="store_true")
    ap.add_argument("--ragas-cases", type=int, default=None, help="score RAGAS on only the first N cases")
    args = ap.parse_args()
    if not config.GROQ_API_KEY:
        sys.exit("GROQ_API_KEY is not set: the eval needs the real LLM, so it was not run (no results were faked).")
    db.init_db()
    modes = [m.strip() for m in args.modes.split(",")]
    if "full" not in modes:
        modes.insert(0, "full")
    summary = evaluate(load_cases(args.limit), modes, args.delay, not args.no_ragas, args.ragas_cases)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "eval_report.md").write_text(report_md(summary))
    print(report_md(summary))
    print(f"Wrote {config.RESULTS_DIR / 'eval_report.md'}")


if __name__ == "__main__":
    main()
