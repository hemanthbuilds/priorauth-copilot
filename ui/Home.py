"""Home page: a simple chatbot for billing staff.

Attach a doctor's note (.txt or .pdf) or paste it, then ask anything, e.g. "Does this note meet the knee
replacement policy?". The bot searches the Medicare policy PDFs (hybrid RAG), the LLM answers from those passages
and the note, and every answer shows where it came from: quotes from the note and the policy (file and page).

Run: uv run streamlit run ui/Home.py
"""

import streamlit as st

from common import api, api_call, badge, banner  # first: it puts the project folder on the import path

from app.guardrails import find_injection  # noqa: E402
from app.intake import find_cpt, find_icd  # noqa: E402
from app.note_reader import NoteReadError, read_note  # noqa: E402

st.set_page_config(page_title="PriorAuth Copilot", page_icon="🩺")
st.title("🩺 PriorAuth Copilot")
banner()

NOTE_MIN_CHARS = 200  # a pasted message this long is treated as a doctor's note, not a question
VERDICTS = {"meets": "✅ **Meets the policy**", "missing_info": "📝 **Missing documentation**",
            "does_not_meet": "❌ **Does not meet the policy as written**"}
WELCOME = ("Hi! 👋 Send me a **doctor's note** (attach a .txt or .pdf with **+**, or paste it), then ask me anything, "
           "for example:\n\n- *Does this note meet the knee replacement policy? Code M17.11*\n"
           "- *What's missing for a lumbar MRI?*\n- *What does the sleep study policy say about home tests?*\n\n"
           "I answer from the Medicare policy PDFs and show you exactly where every answer comes from. "
           "Please don't include patient names; I remove other personal details automatically.")


@st.cache_data(ttl=300)
def procedures() -> dict[str, str]:
    return api("GET", "/api/procedures") or {}


state = st.session_state
for key, default in {"messages": [], "note": "", "note_name": "", "cpt": None, "icd": []}.items():
    state.setdefault(key, default)
procs = procedures()

with st.sidebar:
    if st.button("🆕 New chat"):
        for key in ("messages", "note", "note_name", "cpt", "icd"):
            del state[key]
        st.rerun()
    st.caption(f"📄 Note: {state.note_name or 'none yet'}")
    st.caption("Policies: " + "; ".join(f"{n} ({c})" for c, n in procs.items()))


def show_answer(a: dict, i: int) -> None:
    if a["verdict"] in VERDICTS:
        st.markdown(VERDICTS[a["verdict"]])
    st.markdown(a["answer"])
    if a.get("code_check"):
        st.caption(f"Diagnosis code check: {a['code_check']}")
    if a.get("removed_personal_data"):
        st.caption(f"🔒 For privacy I removed: {', '.join(a['removed_personal_data'])}.")
    if not a["retrieval"]["reranked"]:  # the AI service failed: nothing was searched or quoted
        return
    st.caption(f"Searched: {a['policy']['title'] if a.get('policy') else 'all policies'}")
    with st.expander("📚 Where this came from"):
        for q in a["note_quotes"]:
            st.markdown(f"📝 **Doctor's note:**\n> {q}")
        for q in a["policy_quotes"]:
            st.markdown(f"📜 **Policy** ({q['source']}, page {q['page']}):\n> {q['text']}")
        if not a["note_quotes"] and not a["policy_quotes"]:
            st.caption("No quotes for this answer.")
        if a["unverified_quotes"]:
            st.warning("These quotes were NOT found in the note or policy, so they were left out: "
                       + " / ".join(f"“{q}”" for q in a["unverified_quotes"]))
        passages = a["retrieval"]["reranked"]
        if passages:
            st.caption("Policy passages the search found (best match first):")
            for n, h in enumerate(passages, 1):
                st.text(f"{n}. {h['source']}, page {h['page']} (match {h['score']})\n{h['text'][:500]}")
    if a.get("answer_id") and st.button("📊 Quality scores", key=f"quality_{i}"):
        s, error = api_call("GET", f"/api/answers/{a['answer_id']}/scores")
        if error:
            st.caption(error)
        elif s["status"] == "scoring":
            st.caption("⏳ Still scoring in the background (can take a few minutes). Try again shortly.")
        else:
            st.markdown("".join(badge(m.replace("_", " ").capitalize(), v) for m, v in s["scores"].items()),
                        unsafe_allow_html=True)
            st.caption("RAGAS scores (0 to 1): faithfulness = backed by the sources; answer relevancy = answers "
                       "the question; context precision = the search found the right passages.")
        if s and s.get("trace_url"):
            st.markdown(f"[🔎 View in LangSmith]({s['trace_url']})")


def show(msg: dict, i: int) -> None:
    with st.chat_message(msg["role"]):
        if msg.get("answer"):
            show_answer(msg["answer"], i)
        else:
            st.markdown(msg["text"])


def say(text: str = "", answer: dict | None = None) -> None:
    state.messages.append({"role": "assistant", "text": text, "answer": answer})
    show(state.messages[-1], len(state.messages) - 1)


def small_talk(text: str) -> str | None:
    """Instant replies to greetings and thanks, without using the AI."""
    t = text.lower().strip(" !.?")
    if t in {"hi", "hello", "hey", "hi there", "good morning", "good afternoon"}:
        return "Hello! 😊 Send me a doctor's note and ask me anything about it."
    if t.startswith(("thanks", "thank you")) or t in {"thx", "ty", "great", "perfect", "cool"}:
        return "You're welcome! Ask me anything else, or send another note."
    return None


with st.chat_message("assistant"):
    st.markdown(WELCOME)
for n, m in enumerate(state.messages):
    show(m, n)

prompt = st.chat_input("Ask a question, paste a note, or attach a file", accept_file=True,
                       file_type=["txt", "pdf"], max_upload_size=5)
if prompt:
    text, files = (prompt.text or "").strip(), prompt.files or []
    state.messages.append({"role": "user", "text": (text + "".join(f"\n\n📎 {f.name}" for f in files)).strip()})
    show(state.messages[-1], len(state.messages) - 1)

    # 1. A new note: an attached file, or a long pasted message.
    try:
        if files:
            state.note, state.note_name = read_note(files[0].name, files[0].getvalue()), files[0].name
        elif len(text) >= NOTE_MIN_CHARS:
            state.note, state.note_name, text = text, "pasted note", ""
    except NoteReadError as e:
        say(f"Hmm, I couldn't read that file. {e}")
        st.stop()
    if (files or not text) and state.note and find_injection(state.note):
        say("I can't use this note: it contains text that looks like instructions to the AI ("
            + "; ".join(find_injection(state.note)) + "). Please check the document and send a clean copy.")
        state.note, state.note_name = "", ""
        st.stop()
    if not text:
        say(f"Got the note ({len(state.note.split())} words). Now ask me anything, e.g. *Does this note meet the "
            "knee replacement policy? Code M17.11*")
        st.stop()

    # 2. A question.
    reply = small_talk(text)
    if reply:
        say(reply)
        st.stop()
    state.cpt = find_cpt(text, procs) or state.cpt  # remember the procedure and codes for follow-up questions
    state.icd = find_icd(text) or state.icd
    history = [{"role": m["role"], "content": m["answer"]["answer"] if m.get("answer") else m["text"]}
               for m in state.messages[-7:-1]]
    with st.spinner("Searching the policy and reading the note…"):
        out, error = api_call("POST", "/api/chat", json={"question": text, "note": state.note, "cpt": state.cpt,
                                                         "icd_codes": state.icd, "history": history})
    if out:
        say(answer=out)
    else:
        say(f"Sorry, I couldn't answer that. {error}")
