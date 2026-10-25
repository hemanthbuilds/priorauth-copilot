"""Asks the LLM to draft synthetic test cases (10 per procedure) with expected per-criterion statuses and
an expected outcome. Output: data/cases/cases_draft.json. A human must review it before saving it as
data/cases/cases.json. Every draft label is re-checked against the real decision rules and flagged if
the LLM's outcome doesn't follow from its own statuses.

Run: uv run python -m eval.make_cases
"""

import json
import sys
import time
from typing import Literal

from pydantic import BaseModel

from app import config, graph_store, llm
from app.decision import decide, group_status
from app.ingest.build_graph import load_procedures
from app.models import GroupStatus, LabeledCase, Status
from app.prompts import MAKE_CASES_PROMPT

PER_PROCEDURE = 10
DRAFT_FILE = config.DATA / "cases" / "cases_draft.json"


class ExpectedStatus(BaseModel):
    criterion_id: str
    status: Status


class DraftCase(BaseModel):
    case_id: str
    description: str
    patient_name: str
    patient_age: int
    icd_codes: list[str]
    clinical_notes: str
    expected_outcome: Literal["APPROVE", "REQUEST_INFO", "SEND_TO_CLINICIAN"]
    expected_statuses: list[ExpectedStatus]


class DraftCases(BaseModel):
    cases: list[DraftCase]


def rule_outcome(case: LabeledCase) -> str:
    """What our decision rules say the outcome should be for these labels (assuming high confidence)."""
    from app.models import CriterionResult
    policy = graph_store.get_policy(case.cpt_code)
    groups = graph_store.get_criteria(policy)
    results = [CriterionResult(criterion_id=c.id, criterion_text=c.text, status=case.expected_statuses.get(c.id, "missing"),
                               reasoning="", note_evidence=None, policy_evidence=[], confidence="high")
               for g in groups for c in g.criteria]
    statuses = [GroupStatus(group_id=g.group_id, logic=g.logic, description=g.description,
                            criterion_ids=[c.id for c in g.criteria],
                            status=group_status(g.logic, [case.expected_statuses.get(c.id, "missing") for c in g.criteria]))
                for g in groups]
    return decide(statuses, results, graph_store.diagnosis_supports(case.icd_codes, case.cpt_code)).outcome


def main() -> None:
    if not config.GROQ_API_KEY:
        sys.exit("GROQ_API_KEY is not set: cannot draft cases. Set it in .env and re-run.")
    drafted = []
    for proc in load_procedures():
        policy = graph_store.get_policy(proc["cpt"])
        criteria = [g.model_dump() for g in graph_store.get_criteria(policy)]
        prompt = MAKE_CASES_PROMPT.format(
            cpt_code=proc["cpt"], procedure_name=proc["name"], icd_codes=", ".join(proc["diagnoses"]),
            criteria_json=json.dumps(criteria, indent=1), n=PER_PROCEDURE, id_prefix=f"EVAL-{proc['cpt']}")
        print(f"Drafting {PER_PROCEDURE} cases for CPT {proc['cpt']} ...")
        out = llm.structured(prompt, DraftCases)
        for d in out.cases:
            case = LabeledCase(cpt_code=proc["cpt"], expected_statuses={s.criterion_id: s.status for s in d.expected_statuses},
                               **d.model_dump(exclude={"expected_statuses"}))
            expected_by_rules = rule_outcome(case)
            if expected_by_rules != case.expected_outcome:
                case.description = (f"LABEL CHECK: the LLM said {case.expected_outcome} but the rules give "
                                    f"{expected_by_rules} for these statuses. {case.description}")
            drafted.append(case)
        time.sleep(2)  # be gentle with Groq rate limits
    DRAFT_FILE.write_text(json.dumps([c.model_dump() for c in drafted], indent=2))
    flagged = sum(1 for c in drafted if c.description.startswith("LABEL CHECK"))
    print(f"Wrote {len(drafted)} draft cases to {DRAFT_FILE} ({flagged} flagged for label review).\n"
          f"REVIEW every case and label, then save the good ones as {config.CASES_FILE}.")


if __name__ == "__main__":
    main()
