"""Test setup: fictional practice policies (tests/fixtures), a throwaway database, no tracing, and fakes for the
LLM and retriever so the chatbot tests run offline and give the same answer every time."""

import json
import os
import re
import tempfile

os.environ["DATA_DIR"] = os.path.join(os.path.dirname(__file__), "fixtures")  # fictional practice policies
os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test.db")
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LIVE_EVAL"] = "true"

import pytest  # noqa: E402

from app import config, db  # noqa: E402
from app.models import Hit, LLMAnswer, PolicyQuote, RetrievalTrace, TestCase  # noqa: E402

db.init_db()
CASES = [TestCase.model_validate(c) for c in json.loads(config.CASES_FILE.read_text())]
FAKE_PASSAGE = "Coverage requires advanced arthritis on imaging and failed conservative treatment for 12 weeks."


def case(case_id: str) -> TestCase:
    return next(c for c in CASES if c.case_id == case_id)


class FakeLLM:
    """Answers like a well-behaved LLM: the expected verdict for a known test note, with real quotes.

    `override` forces a specific LLMAnswer (e.g. one with a made-up quote); `fail` makes every call fail.
    """

    def __init__(self):
        self.override: LLMAnswer | None = None
        self.fail = False
        self.calls = 0

    def structured(self, prompt: str, schema, retries: int = 2):
        from app.llm import LLMError
        self.calls += 1
        if self.fail:
            raise LLMError("fake LLM is down")
        assert schema is LLMAnswer
        if self.override:
            return self.override
        note = re.search(r"<doctors_note>\n(.*?)\n</doctors_note>", prompt, re.S).group(1).strip()
        known = next((c for c in CASES if c.clinical_notes[:60] in note), None)
        passage = re.search(r"\[(\S+), page (\d+)\]\n(.+)", prompt)
        policy_quotes = [PolicyQuote(text=" ".join(passage.group(3).split()[:8]), source="made-up.pdf", page=99)] \
            if passage else []
        return LLMAnswer(answer="Fake answer citing the policy.",
                         verdict=known.expected_verdict if known else "not_applicable",
                         note_quotes=[note.split(". ")[0]] if known else [], policy_quotes=policy_quotes)


class FakeRetriever:
    """Returns one policy passage, with a configurable rerank score."""

    def __init__(self, score: float = 5.0):
        self.score = score
        self.last_policy_id = "not called"

    def retrieve(self, query: str, policy_id, mode: str = "full") -> RetrievalTrace:
        self.last_policy_id = policy_id
        hit = Hit(chunk_id="c1", source="total_knee.txt", page=2, text=FAKE_PASSAGE, score=self.score)
        return RetrievalTrace(bm25=[hit], vector=[hit], fused=[hit], reranked=[hit])


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr("app.llm.structured", fake.structured)
    return fake


@pytest.fixture
def fake_retriever(monkeypatch):
    fake = FakeRetriever()
    monkeypatch.setattr("app.retriever.get_retriever", lambda: fake)
    return fake
