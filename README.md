# PriorAuth Copilot

A decision-support assistant for prior authorization reviewers (proof of concept).
**A lookup table decides which policy applies, hybrid search finds what it says, the LLM judges the notes, code makes
the decision, and a human confirms.** The AI never denies: anything that could be a "no" goes to a clinician.

> ⚠️ Decision support only. A human makes the final decision. Use only public policies and **synthetic** patients (no real patient data).

See [HOW_IT_WORKS.md](HOW_IT_WORKS.md) for the concepts, a pipeline diagram and interview Q&A, and [SPEC.md](SPEC.md) for the build spec.

## The business case in one minute

Before an expensive procedure (MRI, knee replacement, sleep study), Medicare and insurers want proof the patient
meets their coverage policy. Today a billing person reads the doctor's note, finds the right policy, and checks
every rule by hand: slow (20–40 min), easy to miss something, and a missed detail means a **denial**, a resubmission
and weeks of delay. This app does the checking in under a minute and says, with quotes from the note and the policy:
**ready to submit**, **what's missing** (so you ask the doctor *before* submitting), or **needs a clinician**.

## Quick start for billing staff

1. Open http://localhost:8501. The home page is a chat.
2. Attach the doctor's note with **+** (a .txt or text-based .pdf) or paste it, and type the procedure and diagnosis
   code, e.g. *"knee replacement, M17.11"* (all in one message or one after the other; it asks for what's missing).
3. You get one answer (ready to submit / what's missing / needs a clinician) and, under **Where this came from**,
   every policy requirement with the sentence from the note and the passage from the Medicare policy (file and section).
4. Ask follow-up questions in the same chat ("why is that missing?", "what if PT lasted 8 weeks?").
   Send a new note, or click **New check** in the sidebar, to start over.

Example notes to try are in `data/sample_notes/` (see its README.txt for the codes to enter).
Scanned PDFs (images with no text) can't be read yet; paste the text instead.

## 1. Setup

Needs [uv](https://docs.astral.sh/uv/) and Python 3.12 or 3.13 (`.python-version` asks for 3.13).

```bash
uv sync
```

```bash
cp .env.example .env
```

Then edit `.env`:

| Variable | Needed? | What it does |
|---|---|---|
| `GROQ_API_KEY` | **Yes** for real reviews | Free key from console.groq.com. Without it every review safely falls back to SEND_TO_CLINICIAN ("automatic check failed"). |
| `GROQ_MODEL` | no | Default `openai/gpt-oss-120b`. The spec asked for `llama-3.3-70b-versatile`, but on a free key it now returns 404 `model_not_found` (Groq lists it as Enterprise). To see which models your key can use: `curl -s https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"`. |
| `USE_SAMPLE_DATA` | no | `true` (default) = the 3 fictional policies in `data/sample_policies/`. `false` = the **real Medicare policies** in `data/policies/` (downloaded by `fetch_cms`, see section 6). |
| `RAGAS_MODEL` | no | The model RAGAS uses as its judge (default: same as `GROQ_MODEL`). `openai/gpt-oss-20b` is recommended: on Groq each model has its own daily token budget, so scoring doesn't eat into the checks, and a different judge is fairer. |
| `LIVE_EVAL` | no | `true` (default) scores every review with RAGAS in the background. If you hit Groq limits, set `RAGAS_MODEL` to a different model first; set `LIVE_EVAL=false` only as a last resort. |
| `MIN_RERANK_SCORE` | no | Below this cross-encoder score a criterion is marked `missing` / low confidence without an LLM call. See section 6. |
| `API_URL` | no | Where the UI finds the API (default `http://localhost:8000`). |
| `LANGSMITH_API_KEY` | optional | Turns on LangSmith tracing (free at smith.langchain.com). Empty = tracing off, app works normally. |

## 2. Build the indexes (ingest)

With the sample data nothing needs downloading. The first run downloads two small local models (~100 MB total) into `data/index/models/`.

```bash
uv run python -m app.ingest.chunk_index
```

To check the policy setup files (every procedure has rules, every rule group has criteria, no duplicates), run
`uv run python -m app.policy_store`. It exits with an error if `criteria.json` is broken; the app also checks on startup.

## 3. Run

API (terminal 1):

```bash
uv run uvicorn app.api:app --port 8000
```

UI (terminal 2):

```bash
uv run streamlit run ui/Home.py
```

Open http://localhost:8501 to check a note. For the sample cases: **Case Queue** page → **Load sample cases** → pick a case → **Run review**. If port 8000 is already taken, start the API with `--port 8010` and set `API_URL=http://localhost:8010` in `.env`.

Try: CASE-001 (should approve), CASE-002 (missing info; then ask "What if PT lasted 8 weeks?"), CASE-004 (fails a rule → clinician), CASE-006 (wrong diagnosis code → clinician).

## 4. Tests

```bash
uv run pytest
```

Tests use a fake LLM and need no API key. The fake answers from the human labels, so test "accuracy" says nothing about the real model.

## 5. Evaluation

```bash
uv run python -m eval.run_eval
```

Runs the labeled cases in `data/cases/cases.json` through three setups (full, all-policies, vector-only), scores RAGAS faithfulness and context recall, and writes `results/eval_report.md` (also shown on the Metrics page). Options: `--limit 3`, `--modes full`, `--delay 3` (more delay if Groq rate-limits you), `--no-ragas`. It refuses to run without `GROQ_API_KEY` rather than produce fake numbers.

To draft 30 more test cases with the LLM:

```bash
uv run python -m eval.make_cases
```

This writes `data/cases/cases_draft.json`. **Review every case and label** (cases marked `LABEL CHECK` have an outcome that doesn't follow from their own statuses), then merge the good ones into `data/cases/cases.json`. Six reviewed sample cases ship already.

## 6. Using the real CMS policies

The real policies come from the **CMS Medicare Coverage Database**, the government's public library of Medicare
coverage rules, through its free API (no account needed). They are already downloaded into `data/policies/`:

| Procedure | Policy (LCD) | Billing article (accepted ICD-10 codes) |
|---|---|---|
| Lumbar MRI, CPT 72148 | L34220 Lumbar MRI (Noridian) | A57206 (3,360 codes) |
| Total knee replacement, CPT 27447 | L36575 Total Knee Arthroplasty (Noridian) | A57685 (1,395 codes) |
| Sleep study, CPT 95810 | L36861 Polysomnography and Other Sleep Studies (Noridian) | A57697 (26 codes) |

The IDs in SPEC.md are out of date: L37281 / A57207 were retired on 10/23/2025, and L34040 / A57698 don't exist.

To refresh them (e.g. after CMS publishes a new version), then rebuild:

```bash
uv run python -m app.ingest.fetch_cms
```
```bash
uv run python -m app.ingest.chunk_index
```

Set `USE_SAMPLE_DATA=false` in `.env` to use them. The checklist for each policy is `data/policies/criteria.json`.
**It was drafted by Claude from the real policy text and should be reviewed by a billing lead or clinician.**
Only each policy's main coverage path is included (for example knee revisions and fractures are not). To have the
LLM draft it instead, run `uv run python -m app.ingest.extract_criteria`, review `criteria_draft.json`, and save it as
`criteria.json`. If a policy changes, re-check the checklist against the new text.

"Page" in citations means a section of the downloaded policy: page 1 = Coverage Indications, page 2 = Associated
Information. On the real policies the top rerank scores range from about 1.2 to 7.3, all above the default
`MIN_RERANK_SCORE` of 0.0.

## 7. LangSmith (optional)

Set `LANGSMITH_API_KEY` (and keep `LANGSMITH_TRACING=true`). Each review then appears as one trace (`review_case` → `extract_facts`, `policy_lookup`, `check_criterion: 1a` → `hybrid_retrieve` + `llm_check`, …, `decide`, `guardrails`), tagged with `mode:` and `procedure:` and with the case ID and outcome in its metadata. Live RAGAS scores and `human_agreed` (1 = reviewer confirmed, 0 = overrode) are attached as feedback, and the Review page shows a **View trace in LangSmith** link. `run_eval.py` uploads the cases as the dataset `priorauth-copilot-cases` and runs each setup as an experiment.

## 8. Project layout

```
app/        config, models, prompts, llm, policy_store, retriever, extractor, checker, decision,
            guardrails, pipeline, chat, actions, db, live_eval, observability, api
app/ingest/ fetch_cms (real policies), chunk_index, extract_criteria
ui/         Home.py (chat for billing staff) + pages: Review Case, Metrics, Case Queue; common.py
eval/       make_cases.py, run_eval.py
data/       config/procedures.yaml, sample_policies/ (+ reviewed criteria.json), policies/ (your PDFs),
            cases/cases.json (6 reviewed cases), index/ (generated), app.db (generated)
tests/      decision, policy lookup, rrf, guardrails, pipeline (fake LLM), api, eval bookkeeping
```

## 9. Choices made where the spec was open

- **One chunk size (~800 characters) instead of parent/child chunks**, to keep the RAG simple: the top 3 reranked chunks are sent straight to the LLM.
- **Sample policies are `.txt` files** with `=== Page N ===` markers instead of PDFs (no PDF-writing library needed). Real policies are read with `PyPDFLoader`; page numbers are 1-based in both.
- **Reviewed criteria live next to their policies** (`data/sample_policies/criteria.json`, `data/policies/criteria.json`), and indexes go to `data/index/sample/` or `data/index/cms/`, so switching `USE_SAMPLE_DATA` never mixes the two.
- **ICD-10 codes for `SUPPORTS` edges come from `procedures.yaml`** (hand-copied from the billing article), not parsed by the LLM. One supporting code on the case is enough.
- **Extra API routes:** `POST /api/cases/load_samples` (Home button), `POST /api/cases/{id}/add_note` ("Add to case" after a what-if), and `action: "finalize"` on `/action` (confirm/override return an editable draft; finalize saves it and updates the status).
- **Low-confidence or LLM-failure results** become `missing` + `low`, which the decision rules send to a clinician. Prompt-injection text is rejected before any LLM call, in notes and in chat; personal identifiers are removed at the API (see HOW_IT_WORKS.md, "Guardrails").
- **Structured output uses Groq's JSON-schema mode for gpt-oss models**: in tool-calling mode `gpt-oss-20b` often called tools that don't exist (0 of 4 test calls worked), JSON-schema mode worked 4 of 4.
- **Faithfulness context = retrieved policy text + the clinical notes**, because each reasoning cites both; scoring against the policy alone would penalize correct answers.
- **Context recall (offline only) uses the criterion's policy wording from `criteria.json` as the reference.**
- **Ablation:** "all policies" checks the same rules but searches every policy; "vector only" searches this policy with the vector top 3 and no BM25, RRF or rerank (and no rerank-score guardrail).
- **No knowledge graph:** procedure → policy → rules is a plain lookup table (`app/policy_store.py`). An earlier version used NetworkX; it gave the same answers with more moving parts.
- `langchain-community` is pinned to 0.4.1 because 0.4.2 removed a module that `ragas` 0.4.3 imports.
- Live RAGAS runs one metric call at a time to stay inside Groq's free-tier limits. Scoring makes several extra Groq calls per checked criterion (faithfulness splits the reasoning into statements and checks each; context precision judges each of the 3 chunks), on top of the review itself.
