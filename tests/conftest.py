"""Test setup: fictional sample data, a throwaway database, no tracing, and fakes for the
LLM and retriever so the pipeline tests run offline and give the same answer every time."""

import json
import os
import re
import tempfile

os.environ["USE_SAMPLE_DATA"] = "true"
os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test.db")
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LIVE_EVAL"] = "true"

import pytest  # noqa: E402

from app import config, db  # noqa: E402
from app.policy_store import load_criteria  # noqa: E402
from app.models import CaseFacts, CheckOutput, Hit, LabeledCase, RetrievalTrace, WhatIfNote  # noqa: E402

db.init_db()

SAMPLE_CASES = [LabeledCase.model_validate(c) for c in json.loads(config.CASES_FILE.read_text())]
CRITERIA = {c.text: (c.id, c.page) for p in load_criteria().policies for g in p.groups for c in g.criteria}


def sample_case(case_id: str) -> LabeledCase:
    return next(c for c in SAMPLE_CASES if c.case_id == case_id)


class FakeLLM:
    """Answers like a well-behaved LLM would, using the human labels in cases.json.

    `overrides` maps criterion id -> CheckOutput to force a specific answer (e.g. a fake quote).
    `fail` makes every call raise, to test the fallback path.
    """

    def __init__(self):
        self.overrides: dict[str, CheckOutput] = {}
        self.fail = False
        self.calls = 0

    def structured(self, prompt: str, schema, retries: int = 2):
        from app.llm import LLMError
        self.calls += 1
        if self.fail:
            raise LLMError("fake LLM is down")
        if schema is CaseFacts:
            return CaseFacts(symptoms=["pain"], symptom_duration_weeks=8)
        if schema is WhatIfNote:
            return WhatIfNote(note_line="Patient completed 8 weeks of physical therapy without relief.")
        if schema is CheckOutput:
            text = re.search(r"<criterion>(.*?)</criterion>", prompt, re.S).group(1).strip()
            cid, _ = CRITERIA[text]
            if cid in self.overrides:
                return self.overrides[cid]
            notes = re.search(r"<clinical_notes>(.*?)</clinical_notes>", prompt, re.S).group(1).strip()
            case = next(c for c in SAMPLE_CASES if c.clinical_notes[:60] in notes)
            status = case.expected_statuses[cid]
            if "8 weeks of physical therapy" in notes and cid == "1a":
                status = "met"
            quote = notes.split(". ")[0] if status != "missing" else None
            return CheckOutput(status=status, reasoning=f"Fake check says {status}.", note_evidence=quote,
                               policy_quote=text[:60], confidence="high")
        raise AssertionError(f"unexpected schema {schema}")

    def text(self, prompt: str, retries: int = 2) -> str:
        self.calls += 1
        if self.fail:
            from app.llm import LLMError
            raise LLMError("fake LLM is down")
        return "Fake answer citing page 2."


class FakeRetriever:
    """Returns the criterion's own policy wording as the only hit, with a configurable rerank score."""

    def __init__(self, score: float = 5.0):
        self.score = score

    def retrieve(self, query: str, policy_id, mode: str = "full") -> RetrievalTrace:
        _, page = CRITERIA.get(query, ("?", 1))
        hit = Hit(chunk_id="c1", source="fake_policy.txt", page=page, text=query, score=self.score)
        return RetrievalTrace(bm25=[hit], vector=[hit], fused=[hit], reranked=[hit])


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr("app.llm.structured", fake.structured)
    monkeypatch.setattr("app.llm.text", fake.text)
    return fake


@pytest.fixture
def fake_retriever(monkeypatch):
    fake = FakeRetriever()
    monkeypatch.setattr("app.retriever.get_retriever", lambda: fake)
    return fake
