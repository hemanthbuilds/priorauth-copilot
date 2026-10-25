# PriorAuth Copilot

A simple chatbot for medical billing staff: give it a doctor's note, ask questions, and it answers from the
real **Medicare coverage policy PDFs** using **hybrid RAG** and an **LLM**, showing exactly where every answer
came from (quotes from the note and the policy, with page numbers).

> ⚠️ Decision support only. A human makes the final decision. Use only synthetic notes (no real patient data).

See [HOW_IT_WORKS.md](HOW_IT_WORKS.md) for a plain-English explanation and interview Q&A.

## How it works

```
your question + doctor's note
   → guardrails (block prompt injection, remove personal details)
   → find the policy (if a procedure is mentioned) + check the diagnosis code against Medicare's list
   → hybrid search of the policy PDF: BM25 + embeddings → RRF fusion → cross-encoder rerank → top 6 passages
   → LLM answers from those passages + the note (verdict, answer, quotes)
   → code checks every quote is really in the note / policy; made-up quotes are removed and flagged
   → answer with sources · RAGAS scores it in the background · LangSmith traces every step
```

The LLM works out the policy's requirements itself from the passages; nothing is hard-coded.

## Setup

Needs [uv](https://docs.astral.sh/uv/) and Python 3.12 or 3.13.

```bash
uv sync
```
```bash
cp .env.example .env
```

| `.env` setting | What it does |
|---|---|
| `GROQ_API_KEY` | **Required.** Free key from console.groq.com |
| `GROQ_MODEL` | The answering model. Default `openai/gpt-oss-120b` (`llama-3.3-70b-versatile` returns 404 on free keys) |
| `RAGAS_MODEL` | The RAGAS judge. `openai/gpt-oss-20b` recommended: a separate daily token budget, and fairer than self-grading |
| `LIVE_EVAL` | `true` scores every answer with RAGAS in the background |
| `LANGSMITH_API_KEY` | Optional. Turns on LangSmith tracing; empty = tracing off, the app works normally |
| `API_URL` | Where the UI finds the API, e.g. `http://localhost:8010` |

Build the search index (first run downloads two small local models, ~100 MB):

```bash
uv run python -m app.ingest.chunk_index
```

## Run

```bash
uv run uvicorn app.api:app --port 8010
```
```bash
uv run streamlit run ui/Home.py
```

Open http://localhost:8501. Attach a note with **+** (.txt or text-based .pdf) or paste it, then ask, e.g.
*"Does this note meet the knee replacement policy? Code M17.11"*. Try the notes in `data/example_notes/`.

## The policies (PDFs)

`data/policies/` holds one PDF per Medicare policy (Noridian, current as of Oct 2026) plus the diagnosis codes
its billing article accepts:

| Procedure | Policy PDF | Billing article (accepted ICD-10 codes) |
|---|---|---|
| Lumbar MRI, CPT 72148 | `L34220.pdf` | A57206 (`L34220_icd10.json`, 3,360 codes) |
| Total knee replacement, CPT 27447 | `L36575.pdf` | A57685 (`L36575_icd10.json`, 1,395 codes) |
| Sleep study, CPT 95810 | `L36861.pdf` | A57697 (`L36861_icd10.json`, 26 codes) |

**To add or update a policy:** download the PDF from the CMS Medicare Coverage Database (open the policy and
click **Download**, e.g. https://www.cms.gov/medicare-coverage-database/view/lcd.aspx?lcdid=36575), put it in
`data/policies/`, add it to `data/procedures.yaml` if it should be linked to a procedure code, then re-run
`chunk_index`. `uv run python -m app.ingest.fetch_cms` can also re-download the three policies and code lists
from the CMS Coverage API. `uv run python -m app.policy_store` checks the setup.

## Evaluation

```bash
uv run python -m eval.run_eval --ragas-cases 3
```

Asks the chatbot "does this note meet the policy?" for each of the 12 labeled cases in `data/test_cases.json`
and reports verdict accuracy, a confusion matrix, a case-by-case table, how often an answer had a made-up quote,
RAGAS faithfulness and answer relevancy, latency, and an ablation (full vs all-policies vs vector-only search).
Cases where the AI service fails (rate limit, network) are reported as errored, not counted as wrong. Writes
`results/eval_report.md`; the Metrics page shows it. It refuses to run without `GROQ_API_KEY` (no fake results).

## Tests

```bash
uv run pytest
```

Runs offline with a fake LLM and fictional practice policies (`tests/fixtures/`, selected with `DATA_DIR`).

## Project layout

```
app/          api, chat (the chatbot), retriever (hybrid search), policy_store, guardrails, prompts, llm,
              note_reader, intake, live_eval (RAGAS), observability (LangSmith), db, models, config
app/ingest/   chunk_index (build the search index), fetch_cms (download policies from the CMS API)
ui/           Home.py (the chat) + pages/1_Metrics.py
eval/         run_eval.py
data/         policies/ (PDFs + accepted codes), procedures.yaml, example_notes/, test_cases.json,
              index/ + app.db (generated)
tests/        chat, guardrails, retrieval, policy lookup, note reader, intake, api, eval; fixtures/
```

## Notes on choices

- **Free Groq tier limits:** 8,000 tokens per minute and 200,000 per day per model. Answers usually take
  3–10 s; when a limit is hit the app waits and retries, then replies "try again in a minute".
- **No rerank threshold:** for broad questions ("does this meet the policy?") the cross-encoder scores even the
  right passages below zero, so the top 6 are always used and RAGAS context precision tracks relevance.
- **Diagnosis codes are checked by code**, against the billing article's list, not by the LLM. If a code isn't
  accepted, the verdict can never be "meets".
- **Quotes may leave words out but never add or change any.** A quote split across two neighbouring chunks
  still verifies. Quotes that fail are removed and shown as unverified.
- **Structured output uses Groq's JSON-schema mode** for gpt-oss models (tool-calling mode failed 4 of 4 times on
  gpt-oss-20b; JSON-schema mode worked 4 of 4).
- `langchain-community` is pinned to 0.4.1 because 0.4.2 removed a module that `ragas` 0.4.3 imports.
