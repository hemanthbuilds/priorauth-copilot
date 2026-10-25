"""Data shapes: retrieved policy passages, the LLM's answer, and what the chat returns to the user."""

from typing import Literal

from pydantic import BaseModel, Field

# The LLM's overall call when the question is about coverage ("not_applicable" for other questions).
Verdict = Literal["meets", "missing_info", "does_not_meet", "not_applicable"]


class Hit(BaseModel):
    """One retrieved policy chunk, with the score from whichever search stage produced it."""
    chunk_id: str
    source: str  # policy file, e.g. L36575.pdf
    page: int
    text: str
    score: float


class RetrievalTrace(BaseModel):
    """Everything the hybrid search did, shown under "passages searched"."""
    bm25: list[Hit] = Field(default_factory=list)
    vector: list[Hit] = Field(default_factory=list)
    fused: list[Hit] = Field(default_factory=list)
    reranked: list[Hit] = Field(default_factory=list)  # the passages the LLM reads


class PolicyInfo(BaseModel):
    policy_key: str
    title: str
    policy_id: str
    source: str


class PolicyQuote(BaseModel):
    text: str = Field(description="Exact words copied from one policy passage, max ~30 words")
    source: str = Field(description="The passage's file name, e.g. L36575.pdf")
    page: int = Field(description="The passage's page number")


class LLMAnswer(BaseModel):
    """What the LLM returns. Code then verifies every quote before the user sees it."""
    answer: str = Field(description="Plain-English answer for a billing specialist, under 200 words")
    verdict: Verdict
    note_quotes: list[str] = Field(description="Exact sentences copied from the doctor's note that support the answer")
    policy_quotes: list[PolicyQuote] = Field(description="Exact policy text that supports the answer")


class ChatAnswer(BaseModel):
    """What the user gets back: the answer plus everything needed to check where it came from."""
    question: str
    answer: str
    verdict: Verdict
    note_quotes: list[str] = Field(default_factory=list)  # verified to be in the note
    policy_quotes: list[PolicyQuote] = Field(default_factory=list)  # verified to be in the retrieved policy text
    unverified_quotes: list[str] = Field(default_factory=list)  # quotes the LLM gave that are NOT in the sources
    policy: PolicyInfo | None = None  # the policy searched (None = searched all policies)
    code_check: str | None = None  # plain fact about the diagnosis codes, from the billing article
    retrieval: RetrievalTrace = Field(default_factory=RetrievalTrace)
    removed_personal_data: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    run_id: str | None = None  # LangSmith run id (if tracing is on)
    answer_id: int | None = None  # SQLite row id once saved


class TestCase(BaseModel):
    """A labeled test case for the evaluation: a note and the verdict a person agreed on."""
    case_id: str
    description: str = ""
    cpt_code: str
    icd_codes: list[str]
    clinical_notes: str
    expected_verdict: Verdict
