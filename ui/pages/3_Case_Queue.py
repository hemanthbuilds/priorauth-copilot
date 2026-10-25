"""Case Queue page: every saved case with its status, a button to load the sample cases, and a form to add a
case by hand. (The simple note checker is the Home page.)"""

import pandas as pd
import streamlit as st

from common import api, banner

st.set_page_config(page_title="PriorAuth Copilot", page_icon="🩺", layout="wide")
st.title("🗂️ Case queue")
banner()

health = api("GET", "/health")
if health:
    if not health["groq_key_set"]:
        st.warning("GROQ_API_KEY is not set on the API server, so every review will fall back to SEND_TO_CLINICIAN.")
    st.caption("Data: Medicare (CMS) policies · "
               f"LangSmith tracing: {'on' if health['tracing'] else 'off'}")

if st.button("Load sample cases"):
    if api("POST", "/api/cases/load_samples"):
        st.success("Sample cases loaded.")

cases = api("GET", "/api/cases") or []
if cases:
    df = pd.DataFrame(cases)[["case_id", "cpt_code", "patient_name", "status", "updated_at"]]
    df.columns = ["ID", "Procedure (CPT)", "Patient (synthetic)", "Status", "Updated"]
    st.dataframe(df, hide_index=True, width="stretch")
    pick = st.selectbox("Open a case", [c["case_id"] for c in cases])
    if st.button("Review this case →", type="primary"):
        st.session_state["case_id"] = pick
        st.switch_page("pages/1_Review_Case.py")
else:
    st.write("No cases yet. Load the samples or add one below.")

with st.expander("➕ New case"):
    with st.form("new_case"):
        c1, c2, c3 = st.columns(3)
        case_id = c1.text_input("Case ID", "CASE-NEW-1")
        name = c2.text_input("Patient name (synthetic only)", "Synthetic Patient")
        age = c3.number_input("Age", 0, 120, 50)
        cpt = c1.selectbox("CPT code", ["72148", "27447", "95810"])
        icd = c2.text_input("ICD-10 codes (comma separated)", "M54.16")
        notes = st.text_area("Clinical notes", height=160)
        if st.form_submit_button("Add case"):
            body = {"case_id": case_id.strip(), "patient_name": name, "patient_age": int(age), "cpt_code": cpt,
                    "icd_codes": [c.strip() for c in icd.split(",") if c.strip()], "clinical_notes": notes}
            if api("POST", "/api/cases", json=body):
                st.success(f"Added {case_id}.")
                st.rerun()
