"""Settings and file paths for the whole app, read once from environment variables (.env)."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes")


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
# The RAGAS "judge" can be a different model: on Groq each model has its own daily token budget, and a
# different judge is fairer than a model grading its own answers.
RAGAS_MODEL = os.getenv("RAGAS_MODEL", GROQ_MODEL).strip()
USE_SAMPLE_DATA = _bool("USE_SAMPLE_DATA", "true")
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "0.0"))
LIVE_EVAL = _bool("LIVE_EVAL", "true")  # background RAGAS scoring; turn off on rate-limited free LLM tiers
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "priorauth-copilot")

DATA = ROOT / "data"
PROCEDURES_YAML = DATA / "config" / "procedures.yaml"
# Labeled test cases: one set for the fictional sample policies, one for the real CMS policies.
CASES_FILE = DATA / "cases" / ("cases.json" if USE_SAMPLE_DATA else "cms_cases.json")
DB_PATH = Path(os.getenv("DB_PATH", str(DATA / "app.db")))
RESULTS_DIR = ROOT / "results"
MODEL_CACHE = DATA / "index" / "models"  # fastembed downloads models here once

# Sample (fictional) and real (CMS) policies get separate indexes so switching modes never mixes them.
if USE_SAMPLE_DATA:
    POLICY_DIR = DATA / "sample_policies"
    INDEX_DIR = DATA / "index" / "sample"
else:
    POLICY_DIR = DATA / "policies"
    INDEX_DIR = DATA / "index" / "cms"

CRITERIA_FILE = POLICY_DIR / "criteria.json"  # human-reviewed rules
CRITERIA_DRAFT_FILE = POLICY_DIR / "criteria_draft.json"  # LLM draft, needs review
GRAPH_FILE = INDEX_DIR / "graph.json"
CHROMA_DIR = INDEX_DIR / "chroma"
BM25_FILE = INDEX_DIR / "bm25.pkl"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

# Retrieval sizes from the spec (section 6).
CHUNK_SIZE, CHUNK_OVERLAP = 800, 100  # one chunk size: big enough to hold a whole rule and its heading
SEARCH_K = 20
RERANK_TOP = 3
RRF_K = 60

# Guardrail limits.
MAX_NOTES_CHARS = 20_000
MIN_NOTES_CHARS = 20
