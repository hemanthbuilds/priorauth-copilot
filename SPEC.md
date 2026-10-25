# PriorAuth Copilot — Graph + Hybrid RAG Decision Assistant (Build Spec)

> **For Claude Code:** Read this whole file first, then build the complete project in one go, following the **Build Order** (section 14). Do not stop to ask questions. Where a library API has changed, check the current docs and use the current recommended way. Keep the code simple and readable, with plain-English comments. Finish by running the checks in section 15.

---

## 1. Problem statement

Health insurers review thousands of **prior authorization** requests every week. A reviewer reads the doctor's request, finds the matching coverage policy, checks every rule against the patient's notes, and decides. This takes 20–40 minutes per case, details get missed, and different reviewers decide differently.

**PriorAuth Copilot** is a conversational assistant for reviewers. For each request it:

1. **Extracts** the key facts (procedure code, diagnosis codes, prior treatments, symptoms) from the doctor's notes.
2. **Walks a knowledge graph** to find exactly which policy and rules apply (graph RAG).
3. **Retrieves** the exact policy wording for each rule with hybrid search (BM25 + vector, RRF fusion, cross-encoder rerank, parent-child chunks).
4. **Checks** each rule against the notes with the LLM: `met`, `not_met`, or `missing`, with evidence quotes from both the notes and the policy.
5. **Decides with code** (deterministic rules, not the LLM): **APPROVE**, **REQUEST_INFO**, or **SEND_TO_CLINICIAN**. The AI never denies.
6. **Acts after human approval**: drafts an approval letter or a missing-information request; the reviewer edits and confirms; the case status updates with a full audit log.
7. **Converses**: the reviewer can ask "Why is rule 2 missing?" or "What if PT lasted 8 weeks?" and the case is re-checked with the new info.

**Design principle:** *the graph decides what applies, hybrid search proves what it says, the LLM judges the notes, code makes the decision, and a human confirms.*

**Data rule:** only public policy documents and **synthetic** patient cases. No real patient data (HIPAA).

---

## 2. Scope (kept small on purpose)

Three procedures, each with one policy:

| Procedure | CPT code(s) | Policy |
|---|---|---|
| MRI of the lumbar spine | 72148 | LCD L37281 "Lumbar MRI" (Noridian JF) + billing article A57207 |
| Total knee replacement (arthroplasty) | 27447 | LCD L36575 "Total Knee Arthroplasty" + billing article A57685 |
| Sleep study (polysomnography) | 95810 | LCD L34040 "Polysomnography and Other Sleep Studies" (Noridian JF) + billing article A57698 |

All three are from the same Medicare contractor region (Noridian, Jurisdiction F), so the policies are consistent. The billing articles list the ICD-10 codes used for `Diagnosis -SUPPORTS-> Procedure` edges.

The README tells the user to download the current policy PDFs from the CMS Medicare Coverage Database and place them in `data/policies/`, and to fill in the exact policy IDs in `data/config/procedures.yaml`. **Also ship `data/sample_policies/`**: three short, clearly labeled **fictional** policies covering the same procedures, so tests and the first demo run without any downloads. `USE_SAMPLE_DATA=true` switches to them.

---

## 3. Tech stack

| Layer | Choice |
|---|---|
| LLM | Groq via `langchain-groq`, model from env `GROQ_MODEL` (default `llama-3.3-70b-versatile`; check Groq's current list and use a current tool-calling model if retired). `temperature=0` |
| Framework | LangChain, plain functions (no agents, no LangGraph) |
| Structured output | Pydantic v2 + `llm.with_structured_output(...)`, with 2 retries on validation errors |
| Knowledge graph | **NetworkX** `DiGraph`, saved to `data/index/graph.json` (node-link format) |
| Graph view | `pyvis`, rendered inside Streamlit |
| PDF loading | `PyPDFLoader` (keep page numbers) |
| Embeddings | `fastembed` with `BAAI/bge-small-en-v1.5` (local, free, no torch) |
| Vector store | Chroma (local persistent) |
| Keyword search | `rank_bm25` |
| Fusion | Reciprocal Rank Fusion, hand-written |
| Reranker | `fastembed` cross-encoder `Xenova/ms-marco-MiniLM-L-6-v2` (local) |
| Evaluation | `ragas` (faithfulness, answer relevancy, context precision, context recall) + custom decision accuracy. Live per-case scoring + offline eval |
| Observability | **LangSmith** tracing (`langsmith` SDK, `@traceable`), with RAGAS scores attached as feedback |
| Storage | SQLite (`data/app.db`): cases, criterion results, actions, audit log |
| Backend | FastAPI |
| Frontend | Streamlit (multi-page) |
| Tooling | `uv`, pinned versions in `pyproject.toml`, `pytest` |

---

## 4. Folder structure

```
priorauth-copilot/
├── README.md
├── SPEC.md                     # this file
├── HOW_IT_WORKS.md             # plain-English concept guide + interview Q&A (section 13)
├── pyproject.toml
├── .env.example
├── data/
│   ├── config/procedures.yaml  # CPT → policy file + policy ID (human-reviewed, deterministic)
│   ├── policies/               # real CMS PDFs (user downloads)
│   ├── sample_policies/        # 3 fictional policies (shipped)
│   ├── cases/                  # synthetic patient requests (JSON)
│   └── index/                  # graph.json, criteria.json, chroma/, bm25.pkl, parents.json
├── app/
│   ├── config.py
│   ├── models.py               # all Pydantic models
│   ├── prompts.py              # all prompts (XML-tag structured role prompting)
│   ├── llm.py                  # Groq client, retries, timeouts
│   ├── ingest/
│   │   ├── chunk_index.py      # parent/child chunks → Chroma + BM25 + parents.json
│   │   ├── extract_criteria.py # LLM extracts rules → criteria_draft.json (human reviews)
│   │   └── build_graph.py      # procedures.yaml + criteria.json → graph.json
│   ├── graph_store.py          # load graph, get_policy(cpt), get_criteria(policy), path explain
│   ├── retriever.py            # bm25, vector, rrf_fuse, rerank, expand_to_parents
│   ├── extractor.py            # notes → CaseFacts
│   ├── checker.py              # one criterion + notes + evidence → CriterionResult
│   ├── decision.py             # deterministic decision rules
│   ├── pipeline.py             # review_case(): runs all steps, times each stage
│   ├── chat.py                 # conversational Q&A + "what if" re-check about a case
│   ├── actions.py              # draft approval letter / info request
│   ├── guardrails.py
│   ├── db.py                   # SQLite tables + audit log
│   ├── live_eval.py            # background RAGAS scoring per review → SQLite + LangSmith feedback
│   ├── observability.py        # LangSmith setup, trace metadata, run URL helper
│   └── api.py                  # FastAPI routes
├── ui/
│   ├── Home.py                 # case queue
│   └── pages/
│       ├── 1_Review_Case.py    # main screen
│       ├── 2_Knowledge_Graph.py
│       └── 3_Metrics.py
├── eval/
│   ├── make_cases.py           # LLM drafts 30 synthetic cases + expected labels → cases_draft.json
│   └── run_eval.py             # decision accuracy, criterion accuracy, RAGAS, latency → results/eval_report.md
└── tests/
    ├── test_decision.py
    ├── test_graph.py
    ├── test_rrf.py
    ├── test_guardrails.py
    ├── test_pipeline_mock.py   # full run with a fake LLM
    └── test_api.py
```

---

## 5. Knowledge graph (graph RAG)

### What's in it

| Node type | Example | Key attributes |
|---|---|---|
| `Procedure` | `CPT:72148` | name |
| `Diagnosis` | `ICD:M54.50` | name |
| `Policy` | `POLICY:lumbar_mri` | title, source file, policy ID |
| `CriteriaGroup` | `GROUP:lumbar_mri:1` | logic (`ALL` or `ANY`), description |
| `Criterion` | `CRIT:lumbar_mri:1a` | text, source page |

| Edge | Meaning |
|---|---|
| `Procedure -COVERED_BY-> Policy` | Which policy governs this procedure |
| `Diagnosis -SUPPORTS-> Procedure` | Diagnoses that can justify the procedure (from the policy) |
| `Policy -REQUIRES-> CriteriaGroup` | Every group must be satisfied |
| `CriteriaGroup -HAS_OPTION-> Criterion` | Inside an `ANY` group, one criterion is enough; inside `ALL`, every one is needed |

### How it's built (reliably)

1. `procedures.yaml` (written by hand) maps each CPT code to its policy file. **Which policy applies is never an LLM guess.**
2. `extract_criteria.py`: the LLM reads each policy and extracts the criteria groups and criteria (with page numbers) into Pydantic models → `criteria_draft.json`.
3. The README tells the user to **review** the draft and save it as `criteria.json`. Ship a ready, reviewed `criteria.json` for the sample policies.
4. `build_graph.py` builds the NetworkX graph from `procedures.yaml` + `criteria.json` and validates it (every procedure has exactly one policy, every policy has ≥1 group, every group has ≥1 criterion). It fails loudly if the graph is invalid.

### How it's used

```python
policy = get_policy("CPT:72148")             # graph walk, exact
groups = get_criteria(policy)                # groups with their criteria and logic
supported = diagnosis_supports(icd_codes, "CPT:72148")   # True/False, shown as a check
```

---

## 6. Hybrid retrieval (per criterion)

- **Chunking:** parents ~2,000 chars (200 overlap), children ~400 chars (50 overlap). Children are indexed; parents are sent to the LLM. Metadata: `policy_id`, `source`, `page`, `parent_id`.
- **Search is filtered to the policy the graph chose** (Chroma metadata filter + BM25 filtered by `policy_id`). This keeps another procedure's policy from leaking in.
- For each criterion: BM25 top 20 + vector top 20 → `rrf_fuse(k=60)` → cross-encoder rerank against the criterion text → top 3 → expand to parents.
- Load all models and indexes once at startup. Run per-criterion retrieval in parallel with a thread pool.

---

## 7. Models (`models.py`)

```python
class CaseRequest(BaseModel):
    case_id: str
    patient_name: str          # synthetic
    patient_age: int
    cpt_code: str
    icd_codes: list[str]
    clinical_notes: str        # free text from the doctor

class CaseFacts(BaseModel):   # extracted by the LLM
    symptoms: list[str]
    symptom_duration_weeks: int | None
    prior_treatments: list[str]       # e.g. "physical therapy 8 weeks"
    red_flags: list[str]              # e.g. "leg numbness"
    test_results: list[str]
    other_relevant: list[str]

class Evidence(BaseModel):
    source: str
    page: int
    quote: str                 # max ~30 words

class CriterionResult(BaseModel):
    criterion_id: str
    criterion_text: str
    status: Literal["met", "not_met", "missing"]
    reasoning: str             # 1–2 plain-English sentences
    note_evidence: str | None  # quote from the clinical notes, or None
    policy_evidence: list[Evidence]
    confidence: Literal["high", "medium", "low"]

class Decision(BaseModel):
    outcome: Literal["APPROVE", "REQUEST_INFO", "SEND_TO_CLINICIAN"]
    reasons: list[str]         # generated by code from the results
    missing_items: list[str]   # what to ask the doctor for
```

---

## 8. Decision logic (`decision.py`), plain code, fully unit-tested

```python
def group_status(group_logic, statuses):
    # ALL group: met only if every criterion is met
    # ANY group: met if any criterion is met
    # not_met if the group can no longer be satisfied; otherwise missing
    ...

def decide(group_statuses, results, diagnosis_supported) -> Decision:
    # 1. Any low-confidence check or unsupported diagnosis → SEND_TO_CLINICIAN
    # 2. Any group not_met → SEND_TO_CLINICIAN   (the AI never denies; a human does)
    # 3. Any group missing → REQUEST_INFO (list what's missing)
    # 4. All groups met → APPROVE
```

Write a comment at the top: *"The LLM judges messy text. This file applies the fixed business rules, so the same results always give the same decision. This makes it consistent, testable, and auditable."*

`test_decision.py` covers every branch, including ANY/ALL groups.

---

## 9. Prompts (`prompts.py`), structured role prompting with XML tags

```python
EXTRACT_PROMPT = """
<role>You extract clinical facts from a doctor's prior authorization notes.</role>
<rules>
- Only record facts written in the notes. Never infer or add facts.
- Convert durations to weeks when clear ("2 months" → 8).
- If something isn't mentioned, leave it empty or null.
</rules>
<notes>{notes}</notes>
"""

CHECK_PROMPT = """
<role>You are a careful utilization review assistant. You check ONE policy
criterion against a patient's notes.</role>

<criterion>{criterion_text}</criterion>
<policy_context>{policy_context}</policy_context>
<case_facts>{facts}</case_facts>
<clinical_notes>{notes}</clinical_notes>

<rules>
- met: the notes clearly show the criterion is satisfied. Quote the notes.
- not_met: the notes clearly show it is NOT satisfied (e.g. PT lasted 2 weeks, 6 required).
- missing: the notes don't say either way. Never guess.
- Use the policy_context to interpret the criterion exactly as written.
- confidence = low if the wording is ambiguous.
</rules>
"""

CHAT_PROMPT = """
<role>You help an insurance reviewer understand one prior authorization case.</role>
<rules>
- Answer only from the case data, criterion results, and policy context below. Cite pages.
- For "what if" questions, say which criterion would change and how; the app re-runs
  the check with the new info.
- Never give medical advice and never change the decision yourself.
</rules>
<case>{case_json}</case>
"""

LETTER_PROMPT = """..."""  # approval letter / info request: short, professional, cites the policy
```

---

## 10. Pipeline, chat, actions, guardrails

**`pipeline.review_case(case)`** → `CaseReview`: facts, policy, graph path (node list for the UI), criterion results, group statuses, decision, retrieved hits per criterion (BM25, vector, fused, reranked), and `timings_ms` per stage. Saves everything to SQLite.

**`chat.py`:**
- Normal questions → answer with `CHAT_PROMPT`.
- "What if …" → the LLM turns the question into an added note line ("Patient completed 8 weeks of PT"), then `review_case` is re-run on a **copy** with that line appended. Shows a before/after decision diff. The original case is unchanged unless the reviewer clicks "Add to case".

**`actions.py`:** after the reviewer clicks **Confirm decision**:
- APPROVE → draft an approval letter.
- REQUEST_INFO → draft a request listing `missing_items`.
- SEND_TO_CLINICIAN → draft a short handoff summary for the clinical reviewer.

The reviewer can edit the text, then **Finalize**: status updates, the letter is saved, and the audit log records who, what, and when. Nothing is "sent"; this is a POC.

**Guardrails (`guardrails.py`):**

| Check | Action |
|---|---|
| CPT code not in `procedures.yaml` | "Not supported yet." No LLM call |
| Empty or very long notes, or prompt-injection phrases in notes or chat | Reject or flag, and log it |
| Top rerank score below `MIN_RERANK_SCORE` for a criterion | That criterion → `missing`, confidence `low` |
| Evidence quote not actually found in the notes (string check) | Downgrade to `missing`, confidence `low` (catches hallucinated quotes) |
| LLM error after retries | Case → SEND_TO_CLINICIAN with reason "automatic check failed" |
| Always | UI banner: "Decision support only. A human makes the final decision." |

---

## 10b. Live RAGAS scoring + LangSmith observability

### Live RAGAS (`live_eval.py`)

After every review (and every chat answer), a FastAPI `BackgroundTasks` job scores it. **It never blocks or slows the answer.**

| Metric | Scored on | Needs ground truth? |
|---|---|---|
| Faithfulness | Each criterion's reasoning vs its retrieved policy context | No |
| Answer relevancy | Chat answers vs the question | No |
| Context precision (reference-free LLM version) | Reranked chunks vs the criterion text | No |

- Case-level score = average over its criteria. Save per-criterion and per-case scores to SQLite table `eval_scores` (case_id, criterion_id, metric, score, timestamp).
- **Context recall** and **decision accuracy** need labels, so they come only from the offline eval (`run_eval.py`). The offline eval also saves its summary to SQLite so the UI can show it.
- If scoring fails (e.g. a rate limit), save `null` and log it. It never breaks the app.

### LangSmith (`observability.py`)

- Enabled by env vars: `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT=priorauth-copilot`. If the key is missing, tracing is off and the app still works (print one warning).
- LangChain/Groq calls are traced automatically. Wrap our own functions with `@traceable` so each review is **one trace with a clear tree**:

```
review_case  (metadata: case_id, cpt_code, outcome)
├── extract_facts            (LLM)
├── graph_lookup             (policy + criteria found)
├── check_criterion: 1a
│   ├── hybrid_retrieve      (bm25 / vector / rrf / rerank counts + top score)
│   └── llm_check            (LLM)
├── check_criterion: 2a ...
├── decide                   (deterministic outcome + reasons)
└── guardrails               (any flags)
```

- Tag runs with `case_id`, `procedure`, `outcome`, and `mode` (`review`, `chat`, `what_if`, `eval`) so they're easy to filter in LangSmith.
- After live RAGAS finishes, attach each score to the trace as **LangSmith feedback** (`client.create_feedback(run_id, key="faithfulness", score=...)`).
- When a reviewer confirms or overrides, attach feedback `key="human_agreed"` (1 or 0). This tracks how often humans agree with the AI.
- `run_eval.py` also uploads the test cases as a **LangSmith dataset** and runs the eval as a LangSmith experiment, so full vs no-graph vs vector-only can be compared side by side in LangSmith.
- Save each review's LangSmith run URL in SQLite so the UI can link to it.

---

## 11. API (`api.py`)

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/cases` | Case queue with statuses |
| POST | `/api/cases` | Add a new case |
| POST | `/api/cases/{id}/review` | Run the pipeline |
| GET | `/api/cases/{id}` | Case + latest review |
| POST | `/api/cases/{id}/chat` | Body `{message, history}` → answer (+ what-if diff) |
| POST | `/api/cases/{id}/action` | Body `{action: "confirm" or "override", outcome?, letter_text?}` |
| GET | `/api/graph` | Graph nodes and edges (for pyvis) |
| GET | `/api/cases/{id}/scores` | Live RAGAS scores for a case (null while still scoring) + LangSmith trace URL |
| GET | `/api/metrics` | Live RAGAS averages + trend, last offline eval summary, latency, outcome counts, override rate |
| GET | `/health` | Health check |

---

## 12. UI (Streamlit)

**Home (case queue):** table of cases (ID, procedure, status: New, Reviewed, Approved, Info requested, With clinician), a "Load sample cases" button, and a "New case" form.

**Review Case (main screen):**
- Left: the case (patient, CPT, ICD, notes) with evidence quotes **highlighted**.
- Center: a **decision card** (big outcome badge + reasons), then a **criteria checklist**: each criterion with a status icon (✅ met, ❌ not met, ❓ missing), reasoning, note quote, and an expandable "policy says" quote with source and page. ANY/ALL groups shown clearly.
- Buttons: **Confirm decision** / **Override** (choose an outcome + reason). Then the editable letter draft → **Finalize**.
- Right: a **chat panel** for questions and what-ifs (shows the before/after diff).
- **Quality badges** on the decision card: faithfulness and context precision (shows "Scoring…" and then the score, polled from `/api/cases/{id}/scores`; green ≥0.8, amber 0.6–0.8, red <0.6), plus a per-criterion faithfulness score in the checklist and a **"View trace in LangSmith"** link.
- "Under the hood" expander: the graph path (CPT → policy → groups → criteria), per-criterion BM25 / vector / fused / reranked tables, and a stage-timing bar chart.

**Knowledge Graph:** a pyvis interactive graph, colored by node type; selecting a procedure highlights its path.

**Metrics (two sections):**
- **Live quality (every review):** cards for average faithfulness, answer relevancy, and context precision; a line chart of scores over time; a table of the lowest-scoring cases with links to their LangSmith traces.
- **Offline eval (last `run_eval.py`):** decision accuracy, criterion accuracy, context recall, a confusion matrix, and the ablation table (full vs no-graph vs vector-only).
- **Operations:** p50/p95 latency and latency by stage, outcome counts, override rate (how often humans disagree with the AI), and a link to the LangSmith project.

---

## 13. Evaluation + HOW_IT_WORKS.md

**Synthetic cases (`eval/make_cases.py`):** the LLM drafts 30 cases (10 per procedure) with expected per-criterion statuses and an expected outcome: about a third each APPROVE, REQUEST_INFO, and SEND_TO_CLINICIAN, including tricky ones (vague durations, an OR-group met through a red flag, a wrong diagnosis code). Output goes to `cases_draft.json`; the README tells the user to **review** it and save it as `data/cases/cases.json`. Ship 6 reviewed sample cases.

**`eval/run_eval.py`** reports:
- **Decision accuracy** (outcome matches the label) and a confusion matrix
- **Criterion accuracy** (status matches the label)
- **Faithfulness** (RAGAS, on reasoning vs retrieved policy context)
- **Retrieval check:** does the reranked top 3 contain the criterion's source page?
- **Ablation:** full pipeline vs **no graph** (vector search over all policies) vs **vector only** (no BM25/RRF/rerank)
- p50/p95 latency per stage

Writes `results/eval_report.md`. Use a small delay and retries for Groq rate limits.

**HOW_IT_WORKS.md:** plain-English sections on prior authorization, knowledge graphs and NetworkX (plus Neo4j/Cypher as the production option), graph RAG vs normal RAG, hybrid search, BM25, embeddings, RRF (worked numeric example), cross-encoder reranking, parent-child chunking, structured output, why code makes the decision, guardrails and the quote check, human-in-the-loop, each eval metric, the ablation results, and "what I'd do in production". Add a Mermaid diagram of the pipeline. End with 25 likely interview questions with short answers.

---

## 14. Build order

1. Skeleton: `pyproject.toml` (pinned), `config.py`, `models.py`, `prompts.py`, `llm.py`, sample policies, `procedures.yaml`, sample `criteria.json`, 6 sample cases.
2. `decision.py` + `test_decision.py` (every branch).
3. Ingest: `build_graph.py` + `graph_store.py` + `test_graph.py`; `chunk_index.py`; `extract_criteria.py`.
4. `retriever.py` + `test_rrf.py`.
5. `extractor.py`, `checker.py`, `guardrails.py`, `pipeline.py` + `test_guardrails.py`, `test_pipeline_mock.py`.
6. `chat.py`, `actions.py`, `db.py`, `observability.py`, `live_eval.py`, `api.py` + `test_api.py`.
7. Streamlit UI (all 4 pages).
8. `eval/make_cases.py`, `eval/run_eval.py`.
9. `README.md` (setup, downloading CMS policies, reviewing drafts, running ingest, API, UI, eval) and `HOW_IT_WORKS.md`.

---

## 15. Done when (acceptance checks)

- [ ] `pytest` passes, with every decision branch covered.
- [ ] With `USE_SAMPLE_DATA=true` and only `GROQ_API_KEY` set, the whole app runs end to end.
- [ ] `build_graph` validates the graph and fails loudly on a broken `criteria.json`.
- [ ] A clear-approval sample case → APPROVE, every criterion ✅ with note and policy quotes.
- [ ] A case with missing info → REQUEST_INFO, listing exactly what's missing.
- [ ] A case that fails a criterion → SEND_TO_CLINICIAN (never "deny").
- [ ] A fake evidence quote is caught by the quote check and downgraded.
- [ ] A what-if ("PT lasted 8 weeks") shows a before/after decision diff without changing the saved case.
- [ ] Confirm → editable letter → Finalize updates the status and the audit log.
- [ ] The Knowledge Graph page shows the graph and highlights a procedure's path.
- [ ] `run_eval.py` writes `results/eval_report.md` with accuracy, faithfulness, latency, and the ablation table.
- [ ] After a review, the faithfulness and context-precision badges change from "Scoring…" to a number without slowing the review.
- [ ] The Metrics page shows live RAGAS averages, the offline eval results, and latency.
- [ ] With LangSmith keys set, each review appears as one trace with the step tree in section 10b, RAGAS scores attached as feedback, and a working "View trace" link.
- [ ] With no LangSmith key, the app runs normally (tracing is just off).

---

## 16. Config (`.env.example`)

```
GROQ_API_KEY=
GROQ_MODEL=llama-3.3-70b-versatile
USE_SAMPLE_DATA=true
MIN_RERANK_SCORE=0.0        # tune after looking at real scores; the README explains how
API_URL=http://localhost:8000
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=          # optional: free at smith.langchain.com; tracing is off if empty
LANGSMITH_PROJECT=priorauth-copilot
```

---

## 17. Code style

- Simple, readable code over clever one-liners. Plain functions, small files.
- A plain-English comment at the top of every file and on any non-obvious block.
- Type hints everywhere.
- Never crash on an LLM or tool error: retry, then fall back safely (SEND_TO_CLINICIAN) with a clear reason.
