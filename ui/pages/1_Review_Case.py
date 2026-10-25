"""Review Case: the main screen. Left = the case with evidence highlighted, center = decision card and
criteria checklist with confirm/override/finalize, right = chat and what-ifs. "Under the hood" shows
the graph path, retrieval tables and stage timings."""

import html
import re

import pandas as pd
import streamlit as st

from common import OUTCOME_COLORS, STATUS_ICONS, api, badge, banner

st.set_page_config(page_title="Review case", page_icon="🩺", layout="wide")
st.title("Review case")
banner()

cases = api("GET", "/api/cases") or []
if not cases:
    st.warning("No cases. Load the sample cases on the Case Queue page first.")
    st.stop()
ids = [c["case_id"] for c in cases]
default = ids.index(st.session_state["case_id"]) if st.session_state.get("case_id") in ids else 0
case_id = st.selectbox("Case", ids, index=default)
st.session_state["case_id"] = case_id

if st.button("▶ Run review", type="primary"):
    with st.spinner("Extracting facts, walking the graph, retrieving policy text, checking each rule…"):
        if api("POST", f"/api/cases/{case_id}/review"):
            st.session_state.pop(f"draft_{case_id}", None)

data = api("GET", f"/api/cases/{case_id}")
if not data:
    st.stop()
case, review = data["case"], data["review"]


def highlight(notes: str, quotes: list[str]) -> str:
    """Escape the notes, then wrap each evidence quote (or each '...' piece of it) in <mark>."""
    out = html.escape(notes)
    for q in quotes:
        for piece in re.split(r"\.\.\.|…", q):
            words = html.escape(piece.strip()).split()
            if len(words) < 2:
                continue
            pattern = r"\s+".join(re.escape(w) for w in words)
            out = re.sub(f"({pattern})", r"<mark>\1</mark>", out, count=1, flags=re.I)
    return out.replace("\n", "<br>")


left, center, right = st.columns([1.1, 1.7, 1.1], gap="large")

# ---------------- Left: the case ----------------
with left:
    st.subheader(case["patient_name"])
    st.write(f"**Age:** {case['patient_age']} · **CPT:** {case['cpt_code']} · **ICD-10:** {', '.join(case['icd_codes'])}")
    st.caption(f"Status: **{data['status']}**")
    quotes = [r["note_evidence"] for r in (review or {}).get("results", []) if r.get("note_evidence")]
    st.markdown(f"<div style='border:1px solid #ddd;border-radius:8px;padding:10px;line-height:1.5'>"
                f"{highlight(case['clinical_notes'], quotes)}</div>", unsafe_allow_html=True)
    if data["audit"]:
        with st.expander("Audit log"):
            st.dataframe(pd.DataFrame(data["audit"]), hide_index=True)

if not review:
    center.info("Not reviewed yet. Click **Run review**.")
    st.stop()

# ---------------- Center: decision + checklist + actions ----------------
with center:
    d = review["decision"]
    color = OUTCOME_COLORS[d["outcome"]]
    st.markdown(f"<div style='background:{color};color:white;padding:14px;border-radius:10px;font-size:1.6em;"
                f"font-weight:700;text-align:center'>{d['outcome'].replace('_', ' ')}</div>", unsafe_allow_html=True)

    @st.fragment(run_every=3)
    def quality_badges():
        """Polls the live RAGAS scores. Stops refreshing visibly once they're in."""
        s = api("GET", f"/api/cases/{case_id}/scores") or {}
        pending = s.get("status") == "scoring"
        html_badges = badge("Faithfulness", s.get("case", {}).get("faithfulness"), pending) + \
            badge("Context precision", s.get("case", {}).get("context_precision"), pending)
        if s.get("trace_url"):
            html_badges += f" <a href='{s['trace_url']}' target='_blank'>View trace in LangSmith ↗</a>"
        st.markdown(html_badges, unsafe_allow_html=True)
        st.session_state[f"scores_{case_id}"] = s.get("criteria", {})

    quality_badges()
    for reason in d["reasons"]:
        st.write(f"• {reason}")
    if d["missing_items"]:
        st.write("**Ask the doctor for:**")
        for item in d["missing_items"]:
            st.write(f"- {item}")
    if review.get("guardrail_flags"):
        with st.expander(f"⚠️ Guardrail flags ({len(review['guardrail_flags'])})"):
            for f in review["guardrail_flags"]:
                st.write(f"- {f}")
    if review["policy"]:
        st.caption(f"Policy: {review['policy']['title']} · Diagnosis supports procedure: "
                   f"{'✅ yes' if review['diagnosis_supported'] else '❌ no'}")

    st.subheader("Criteria checklist")
    results = {r["criterion_id"]: r for r in review["results"]}
    crit_scores = st.session_state.get(f"scores_{case_id}", {})
    for g in review["group_statuses"]:
        need = "ALL required" if g["logic"] == "ALL" else "ANY one is enough"
        st.markdown(f"**Group {g['group_id']}: {g['description']}** · _{need}_ → {STATUS_ICONS[g['status']]} "
                    f"`{g['status']}`")
        for cid in g["criterion_ids"]:
            r = results[cid]
            faith = crit_scores.get(cid, {}).get("faithfulness")
            with st.container(border=True):
                st.markdown(f"{STATUS_ICONS[r['status']]} **{cid}** {r['criterion_text']}")
                st.markdown(f"{r['reasoning']}  \n" + badge("faithfulness", faith) +
                            f"<span style='font-size:0.85em'>confidence: {r['confidence']}</span>",
                            unsafe_allow_html=True)
                if r["note_evidence"]:
                    st.markdown(f"> 📝 “{r['note_evidence']}”")
                if r["policy_evidence"]:
                    with st.expander("Policy says"):
                        for e in r["policy_evidence"]:
                            st.markdown(f"“{e['quote']}”  \n_{e['source']}, page {e['page']}_")
                for f in r.get("flags", []):
                    st.caption(f"⚠️ {f}")

    st.subheader("Your decision")
    draft_key = f"draft_{case_id}"
    c1, c2 = st.columns(2)
    if c1.button("✔ Confirm decision", width="stretch"):
        out = api("POST", f"/api/cases/{case_id}/action", json={"action": "confirm"})
        if out:
            st.session_state[draft_key] = out
    with c2.popover("✎ Override", width="stretch"):
        new_outcome = st.selectbox("Outcome", ["APPROVE", "REQUEST_INFO", "SEND_TO_CLINICIAN"])
        reason = st.text_input("Reason (required)")
        if st.button("Apply override"):
            out = api("POST", f"/api/cases/{case_id}/action",
                      json={"action": "override", "outcome": new_outcome, "reason": reason})
            if out:
                st.session_state[draft_key] = out
    if draft_key in st.session_state:
        draft = st.session_state[draft_key]
        st.write(f"Draft for **{draft['final_outcome']}** (edit before finalizing; nothing is sent):")
        text = st.text_area("Letter", draft["letter_text"], height=220, key=f"letter_{case_id}")
        if st.button("Finalize", type="primary"):
            out = api("POST", f"/api/cases/{case_id}/action", json={"action": "finalize", "letter_text": text})
            if out:
                st.success(f"Finalized. Case status: {out['status']}")
                st.session_state.pop(draft_key)
                st.rerun()

# ---------------- Right: chat ----------------
with right:
    st.subheader("Ask about this case")
    st.caption('Try: "Why is 1a missing?" or "What if PT lasted 8 weeks?"')
    hist_key = f"chat_{case_id}"
    history = st.session_state.setdefault(hist_key, [])
    with st.form(f"chat_form_{case_id}", clear_on_submit=True):
        msg = st.text_input("Question")
        sent = st.form_submit_button("Send")
    if sent and msg.strip():
        out = api("POST", f"/api/cases/{case_id}/chat",
                  json={"message": msg, "history": [{"role": h["role"], "content": h["content"]} for h in history]})
        if out:
            history.append({"role": "user", "content": msg})
            history.append({"role": "assistant", "content": out["answer"], "what_if": out.get("what_if")})
    for i, h in enumerate(reversed(history)):
        with st.chat_message(h["role"]):
            st.write(h["content"])
            wi = h.get("what_if")
            if wi:
                st.markdown(f"**Before:** `{wi['before']}` → **After:** `{wi['after']}`")
                if wi["changed"]:
                    st.dataframe(pd.DataFrame(wi["changed"]), hide_index=True)
                if st.button("Add to case", key=f"add_{case_id}_{i}"):
                    if api("POST", f"/api/cases/{case_id}/add_note", json={"note_line": wi["note_line"]}):
                        st.success("Added to the notes. Run the review again to update the decision.")

# ---------------- Under the hood ----------------
with st.expander("🔧 Under the hood"):
    st.write("**Graph path:** " + " → ".join(review["graph_path"]))
    for cid, tr in review["retrieval"].items():
        st.markdown(f"**Criterion {cid}**")
        tabs = st.tabs(["BM25", "Vector", "Fused (RRF)", "Reranked (top 3, sent to the AI)"])
        for tab, key in zip(tabs, ["bm25", "vector", "fused", "reranked"]):
            rows = [{"score": h["score"], "source": h["source"], "page": h["page"], "text": h["text"][:160]}
                    for h in tr[key]]
            if rows:
                tab.dataframe(pd.DataFrame(rows), hide_index=True)
            else:
                tab.caption("(not used in this mode)")
    timings = {k: v for k, v in review["timings_ms"].items() if k != "total"}
    st.write(f"**Stage timings (ms)**, total {review['timings_ms'].get('total')} ms")
    st.bar_chart(pd.Series(timings, name="ms"))
