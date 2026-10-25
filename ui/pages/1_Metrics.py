"""Metrics page: live RAGAS quality of every chat answer, the last offline evaluation (verdict accuracy,
confusion matrix, case by case, ablation), and operations (answers, latency, verdicts, LangSmith link)."""

import pandas as pd
import streamlit as st

from common import api, banner

st.set_page_config(page_title="Metrics", page_icon="📊", layout="wide")
st.title("📊 Metrics")
banner()

m = api("GET", "/api/metrics")
if not m:
    st.stop()


def fmt(v, pct=False):
    return "n/a" if v is None else (f"{v:.0%}" if pct else f"{v:.2f}")


st.header("Live quality (RAGAS, every answer)")
c1, c2, c3 = st.columns(3)
c1.metric("Faithfulness", fmt(m["live"]["faithfulness"]), help="Is the answer backed by the policy and note?")
c2.metric("Answer relevancy", fmt(m["live"]["answer_relevancy"]), help="Does the answer address the question?")
c3.metric("Context precision", fmt(m["live"]["context_precision"]), help="Did search find the right passages?")
if m["trend"]:
    trend = pd.DataFrame(m["trend"])
    trend["timestamp"] = pd.to_datetime(trend["timestamp"])
    st.line_chart(trend.pivot_table(index="timestamp", columns="metric", values="score"))
else:
    st.caption("No scores yet. They appear a few minutes after each answer.")

st.header("Offline evaluation (last run of eval/run_eval.py)")
off = m["offline"]
if not off:
    st.caption("No evaluation yet. Run: uv run python -m eval.run_eval")
else:
    full = off["full"]
    st.caption(f"Run at {off['ts']} · {off['n_cases']} labeled cases · model {off['model']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Verdict accuracy", fmt(full["verdict_accuracy"], True))
    c2.metric("Faithfulness", fmt(full.get("faithfulness")))
    c3.metric("Unverified quotes", fmt(full["unverified_quote_rate"], True), help="Share of answers with a made-up quote")
    st.write("**Case by case**")
    st.dataframe(pd.DataFrame(off["cases"]), hide_index=True)
    st.write("**Confusion matrix** (rows = expected, columns = answered)")
    st.dataframe(pd.DataFrame(off["confusion"]).T.fillna(0).astype(int))
    st.write("**Ablation** (same cases, different search setups)")
    st.dataframe(pd.DataFrame(off["ablation"]), hide_index=True)

st.header("Operations")
ops = m["ops"]
c1, c2, c3 = st.columns(3)
c1.metric("Answers", ops["answers"])
c2.metric("Latency p50", f"{ops['latency_p50_ms'] / 1000:.1f} s" if ops["latency_p50_ms"] else "n/a")
c3.metric("Latency p95", f"{ops['latency_p95_ms'] / 1000:.1f} s" if ops["latency_p95_ms"] else "n/a")
if ops["verdicts"]:
    st.bar_chart(pd.Series(ops["verdicts"], name="answers"))
if ops["langsmith_project_url"]:
    st.link_button("Open LangSmith project ↗", ops["langsmith_project_url"])
