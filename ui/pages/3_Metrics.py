"""Metrics page: live RAGAS quality from every review, the last offline eval (accuracy, confusion matrix,
ablation), and operations (latency, outcomes, how often humans override the AI)."""

import pandas as pd
import streamlit as st

from common import api, banner

st.set_page_config(page_title="Metrics", page_icon="📊", layout="wide")
st.title("Metrics")
banner()

m = api("GET", "/api/metrics")
if not m:
    st.stop()


def fmt(v, pct=False):
    if v is None:
        return "n/a"
    return f"{v:.0%}" if pct else f"{v:.2f}"


# ---------------- Live quality ----------------
st.header("Live quality (every review)")
c1, c2, c3 = st.columns(3)
c1.metric("Avg faithfulness", fmt(m["live"]["faithfulness"]))
c2.metric("Avg answer relevancy (chat)", fmt(m["live"]["answer_relevancy"]))
c3.metric("Avg context precision", fmt(m["live"]["context_precision"]))
if m["trend"]:
    trend = pd.DataFrame(m["trend"])
    trend["timestamp"] = pd.to_datetime(trend["timestamp"])
    st.line_chart(trend.pivot_table(index="timestamp", columns="metric", values="score"))
else:
    st.caption("No live scores yet. They appear a little after each review (they need GROQ_API_KEY).")
if m["lowest"]:
    st.write("**Lowest-faithfulness reviews**")
    st.dataframe(pd.DataFrame(m["lowest"]), hide_index=True,
                 column_config={"trace_url": st.column_config.LinkColumn("LangSmith trace")})

# ---------------- Offline eval ----------------
st.header("Offline eval (last run_eval.py)")
off = m["offline"]
if not off:
    st.caption("No offline eval yet. Run: uv run python -m eval.run_eval")
else:
    full = off["full"]
    st.caption(f"Run at {off['ts']} · {off['n_cases']} labeled cases · model {off['model']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Decision accuracy", fmt(full["decision_accuracy"], True))
    c2.metric("Criterion accuracy", fmt(full["criterion_accuracy"], True))
    c3.metric("Context recall", fmt(full["context_recall"]))
    c4.metric("Faithfulness", fmt(full["faithfulness"]))
    st.write("**Confusion matrix** (rows = expected, columns = predicted)")
    st.dataframe(pd.DataFrame(off["confusion"]).T.fillna(0).astype(int))
    st.write("**Ablation**")
    st.dataframe(pd.DataFrame(off["ablation"]), hide_index=True)

# ---------------- Operations ----------------
st.header("Operations")
ops = m["ops"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews", ops["reviews"])
c2.metric("Latency p50", f"{ops['latency_p50_ms']:.0f} ms" if ops["latency_p50_ms"] else "n/a")
c3.metric("Latency p95", f"{ops['latency_p95_ms']:.0f} ms" if ops["latency_p95_ms"] else "n/a")
c4.metric("Override rate", fmt(ops["override_rate"], True), help="How often reviewers disagreed with the AI")
if ops["by_stage"]:
    st.write("**Latency by stage (ms)**")
    st.bar_chart(pd.DataFrame(ops["by_stage"]).T)
if ops["outcomes"]:
    st.write("**Outcome counts**")
    st.bar_chart(pd.Series(ops["outcomes"], name="reviews"))
if ops["langsmith_project_url"]:
    st.link_button("Open LangSmith project ↗", ops["langsmith_project_url"])
else:
    st.caption("LangSmith tracing is off (no LANGSMITH_API_KEY).")
