"""Step 1 of a review: the LLM pulls structured facts (symptoms, durations, treatments, red flags) from the notes."""

from langsmith import traceable

from app import llm
from app.models import CaseFacts, CaseRequest
from app.prompts import EXTRACT_PROMPT


@traceable(name="extract_facts", run_type="chain")
def extract_facts(case: CaseRequest) -> CaseFacts:
    return llm.structured(EXTRACT_PROMPT.format(notes=case.clinical_notes), CaseFacts)
