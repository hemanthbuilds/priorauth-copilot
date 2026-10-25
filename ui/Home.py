"""Home page: a simple chat for billing staff.

Send a doctor's note (attach a .txt/.pdf or paste it) and say the procedure and diagnosis code, e.g.
"knee replacement, M17.11". The assistant checks the note against the Medicare policy and answers: ready to submit,
what's missing, or why it needs a clinician. Every answer shows where it came from: the sentence in the doctor's
note and the passage of the policy (file and section). After that, ask follow-up questions in the same chat.

Run: uv run streamlit run ui/Home.py
"""

from datetime import datetime

import streamlit as st

from common import api, api_call, badge, banner  # first: it puts the project folder on the import path

from app.guardrails import find_injection  # noqa: E402
from app.intake import find_cpt, find_icd  # noqa: E402
from app.note_reader import NoteReadError, read_note  # noqa: E402

st.set_page_config(page_title="PriorAuth Copilot", page_icon="🩺")
st.title("🩺 PriorAuth Copilot")
banner()

WELCOME = ("Hi there! 👋 I help you check whether a doctor's note supports a prior authorization request "
           "before you submit it.\n\n"
           "**To start**, send me the **doctor's note** (attach a .txt or .pdf with the **+** button, or paste the "
           "text) and tell me the **procedure** and **diagnosis code**, for example *“knee replacement, M17.11”*.\n\n"
           "I'll check it against the Medicare policy and show you exactly where every answer comes from. "
           "Please don't include patient names; I remove other personal details automatically.")
HELP = ("I can check a doctor's note against the Medicare coverage policy for **{procs}**.\n\n"
        "1. Send the note (attach or paste it).\n2. Tell me the procedure and diagnosis code.\n"
        "3. I'll tell you if it's ready to submit, what's missing, or why it needs a clinician, with the exact "
        "sentences from the note and the policy.\n\nAfter that, ask me anything about the result.")


def small_talk(text: str) -> str | None:
    """Friendly replies to greetings and thanks, answered instantly without using the AI."""
    t = text.lower().strip(" !.?")
    if len(t) > 60:
        return None
    if t in {"hi", "hello", "hey", "hiya", "good morning", "good afternoon", "good evening", "hi there", "hello there"}:
        return "Hello! 😊 Send me a doctor's note whenever you're ready, plus the procedure and diagnosis code."
    if t.startswith(("thanks", "thank you")) or t in {"thx", "ty", "great", "perfect", "great thanks", "cool"}:
        return "You're welcome! Send another note any time, or ask me more about this one."
    if t in {"help", "what can you do", "how does this work", "how do i use this", "what do you do"}:
        return HELP.format(procs=", ".join(procs.values()))
    if t in {"bye", "goodbye", "see you", "that's all"}:
        return "Bye for now! 👋 Your checks are saved in the Case Queue."
    return None
NOTE_MIN_CHARS = 200  # a pasted message this long is treated as the doctor's note, not a question
LEAD_IN_CHARS = 80  # codes typed before a pasted note ("lumbar MRI, M54.50. <note>") are read from this much


@st.cache_data(ttl=300)
def procedures() -> dict[str, str]:
    return api("GET", "/api/procedures") or {}


state = st.session_state
state.setdefault("messages", [])
state.setdefault("pending", {"cpt": None, "icd": [], "notes": None})
state.setdefault("case_id", None)
procs = procedures()

with st.sidebar:
    if st.button("🆕 New check"):
        state.messages, state.pending, state.case_id = [], {"cpt": None, "icd": [], "notes": None}, None
        st.rerun()
    st.caption("Procedures I know: " + "; ".join(f"{c} {n}" for c, n in procs.items()))


# ---------- How an answer with sources is drawn ----------

def next_steps(review: dict, case: dict) -> tuple[str, list[str]]:
    d, results = review["decision"], {r["criterion_id"]: r for r in review["results"]}
    policy = review["policy"]["title"] if review.get("policy") else "the policy"
    if d["outcome"] == "APPROVE":
        return "✅ **Good news: this note looks ready to submit.** It documents everything the policy asks for.", \
            ["Submit the prior authorization request with this note attached."]
    if d["outcome"] == "REQUEST_INFO":
        return "📝 **Almost there: a few things are missing from the note.** Getting them now avoids a denial later.", \
            ["Before submitting, ask the doctor's office to add:"] + \
            [f"- {m.split(': ', 1)[-1]}" for m in d["missing_items"]]
    steps = []
    if not review.get("diagnosis_supported", True):
        steps.append(f"The diagnosis code(s) {', '.join(case['icd_codes'])} are not on the accepted list for CPT "
                     f"{case['cpt_code']} in {policy}. Check the coding with the provider.")
    for g in review.get("group_statuses", []):
        if g["status"] == "not_met":
            failed = [results[c]["criterion_text"] for c in g["criterion_ids"] if results[c]["status"] == "not_met"]
            steps.append(f"Not met: {g['description']}. " + " / ".join(failed))
    if any(r.startswith("Automatic check failed") for r in d["reasons"]):
        steps.append("Part of the automatic check failed: the AI service was busy, out of its daily limit, or the "
                     "internet connection dropped. Nothing is wrong with the note itself. Send it again in a minute.")
    else:
        steps += [f"Unclear, have someone check: {r['criterion_text']}" for r in review["results"]
                  if r["confidence"] == "low"]
    return "🩺 **This one needs a closer look before you submit.**", steps or d["reasons"]


def show_review(review: dict, case: dict) -> None:
    headline, steps = next_steps(review, case)
    st.markdown(headline)
    st.markdown("**What to do next**\n\n" + "\n".join(s if s.startswith("- ") else f"\n{s}" for s in steps))
    if review.get("policy"):
        st.caption(f"Checked against {review['policy']['title']}")
    st.markdown("**Where this came from**")
    icons = {"met": "✅", "not_met": "❌", "missing": "❓"}
    for r in review["results"]:
        with st.expander(f"{icons[r['status']]} {r['criterion_text']}"):
            st.markdown(f"**Decision:** {r['status'].replace('_', ' ')} ({r['confidence']} confidence). {r['reasoning']}")
            if r["note_evidence"]:
                st.markdown(f"📝 **From the doctor's note:**\n> {r['note_evidence']}")
            else:
                st.markdown("📝 **From the doctor's note:** nothing found about this.")
            for e in r["policy_evidence"]:
                st.markdown(f"📜 **From the policy** ({e['source']}, section {e['page']}):\n> {e['quote']}")
            passages = review.get("retrieval", {}).get(r["criterion_id"], {}).get("reranked", [])
            if passages:
                st.caption("Policy passages the search found for this rule (best match first):")
                for i, h in enumerate(passages, 1):
                    st.text(f"{i}. {h['source']}, section {h['page']} (match score {h['score']})\n{h['text'][:600]}")


def show_quality(case_id: str) -> None:
    """Live RAGAS scores for this check (scored in the background) and its LangSmith trace."""
    s, error = api_call("GET", f"/api/cases/{case_id}/scores")
    if error or not s:
        st.caption(f"Couldn't load quality scores: {error}")
    elif s["status"] == "scoring":
        st.caption("⏳ Still scoring in the background (this can take a few minutes). Try again shortly.")
    else:
        st.markdown(badge("Faithfulness", s["case"].get("faithfulness")) +
                    badge("Context precision", s["case"].get("context_precision")), unsafe_allow_html=True)
        st.caption("Faithfulness: is the reasoning backed by the note and policy? Context precision: did the search "
                   "find the right policy passages? (RAGAS scores, 0 to 1, higher is better)")
    if s and s.get("trace_url"):
        st.markdown(f"[🔎 View this check in LangSmith]({s['trace_url']})")


def show(msg: dict, i: int) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])
        if msg.get("review"):
            show_review(msg["review"], msg["case"])
            if st.button("📊 Quality scores", key=f"quality_{i}"):
                show_quality(msg["case"]["case_id"])
        if msg.get("what_if"):
            w = msg["what_if"]
            st.markdown(f"**Before:** {w['before']} → **After:** {w['after']}")


# ---------- The conversation ----------

with st.chat_message("assistant"):
    st.markdown(WELCOME)
for i, m in enumerate(state.messages):
    show(m, i)


def say(text: str, **extra) -> None:
    msg = {"role": "assistant", "text": text, **extra}
    state.messages.append(msg)
    show(msg, len(state.messages) - 1)


def run_check() -> None:
    p = state.pending
    case = {"case_id": f"NOTE-{datetime.now():%Y%m%d-%H%M%S}", "patient_name": "(not entered)", "patient_age": 0,
            "cpt_code": p["cpt"], "icd_codes": p["icd"], "clinical_notes": p["notes"]}
    with st.spinner("Reading the note and checking it against the policy (usually under a minute)…"):
        added, error = api_call("POST", "/api/cases", json=case)
        out, error = api_call("POST", f"/api/cases/{case['case_id']}/review") if added else (None, error)
    state.pending = {"cpt": None, "icd": [], "notes": None}
    if error:
        say(f"Sorry, I couldn't check that note. {error}\n\nYou can send the note again whenever you're ready.")
        return
    state.case_id = case["case_id"]
    removed = f"🔒 For privacy I removed: {', '.join(added['removed'])}.\n\n" if added.get("removed") else ""
    say(f"{removed}Here's what I found for **{procs[p['cpt']]} (CPT {p['cpt']})** with diagnosis "
        f"**{', '.join(p['icd'])}**. Open any requirement below to see exactly where the answer came from. "
        f"You can also ask me things like *“why is that missing?”* or *“what if PT lasted 8 weeks?”*",
        review=out["review"], case=case)


prompt = st.chat_input("Paste a note, attach a file, or ask a question", accept_file=True, file_type=["txt", "pdf"],
                       max_upload_size=5)  # MB
if prompt:
    text = (prompt.text or "").strip()
    files = prompt.files or []
    user = {"role": "user", "text": (text + "".join(f"\n\n📎 {f.name}" for f in files)).strip() or "(file)"}
    state.messages.append(user)
    show(user, len(state.messages) - 1)

    note = None
    try:
        if files:
            note = read_note(files[0].name, files[0].getvalue())
        elif len(text) >= NOTE_MIN_CHARS:
            note = text
    except NoteReadError as e:
        say(f"Hmm, I couldn't read that file. {e}")
        st.stop()

    reply = small_talk(text) if not files else None
    if reply:
        say(reply)
        st.stop()

    if note is None and state.case_id:
        # After a check, anything that isn't a new note is a follow-up question about that case.
        with st.spinner("Thinking…"):
            out, error = api_call("POST", f"/api/cases/{state.case_id}/chat",
                                  json={"message": text, "history": [{"role": m["role"], "content": m["text"]}
                                                                     for m in state.messages[-6:]]})
        say(out["answer"] if out else error, what_if=out.get("what_if") if out else None)
        st.stop()

    if note and find_injection(note):
        # The API rejects this too; saying it now saves the user a round trip.
        say("I can't use this note: it contains text that looks like instructions to the AI ("
            + "; ".join(find_injection(note)) + "). For safety I won't send it to the AI. "
            "Please check the document and send a clean copy.")
        st.stop()

    p = state.pending
    if note:
        p["notes"] = note
        state.case_id = None  # a new note starts a new check
    # Codes come only from what the user typed: the message next to an attached file, or the start of a pasted
    # note. They are never guessed from the body of the note.
    typed = text[:LEAD_IN_CHARS] if note and not files else text
    p["cpt"] = find_cpt(typed, procs) or p["cpt"]
    p["icd"] = find_icd(typed) or p["icd"]

    missing = []
    if not p["notes"]:
        missing.append("the **doctor's note** (attach a .txt/.pdf or paste it)")
    if not p["cpt"]:
        missing.append("the **procedure**, e.g. *knee replacement*, *lumbar MRI* or *sleep study* (or the CPT code)")
    if not p["icd"]:
        missing.append("the **diagnosis code(s)**, e.g. *M17.11*")
    if missing:
        got = [x for x in [p["cpt"] and f"procedure {p['cpt']} ({procs[p['cpt']]})",
                           p["icd"] and f"diagnosis {', '.join(p['icd'])}", p["notes"] and "the note"] if x]
        thanks = ("Thanks, I've got " + ", ".join(got) + ". " if got else "")
        say(thanks + "To check it, I still need " + " and ".join(missing) + ".")
    else:
        run_check()
