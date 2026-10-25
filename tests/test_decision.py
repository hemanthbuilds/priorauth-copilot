"""Covers every branch of the deterministic decision rules, including ANY and ALL groups."""

import pytest

from app.decision import decide, fallback_decision, group_status
from app.models import CriterionResult, GroupStatus


def result(cid, status, confidence="high"):
    return CriterionResult(criterion_id=cid, criterion_text=f"text {cid}", status=status, reasoning="r",
                           note_evidence=None, policy_evidence=[], confidence=confidence)


def group(gid, logic, ids, status):
    return GroupStatus(group_id=gid, logic=logic, description=f"group {gid}", criterion_ids=ids, status=status)


@pytest.mark.parametrize("statuses,expected", [
    (["met", "met"], "met"),
    (["met", "not_met"], "not_met"),
    (["met", "missing"], "missing"),
    (["missing", "not_met"], "not_met"),
    (["missing"], "missing"),
])
def test_all_group(statuses, expected):
    assert group_status("ALL", statuses) == expected


@pytest.mark.parametrize("statuses,expected", [
    (["met", "not_met"], "met"),
    (["missing", "met"], "met"),
    (["not_met", "not_met"], "not_met"),
    (["not_met", "missing"], "missing"),
    (["missing", "missing"], "missing"),
])
def test_any_group(statuses, expected):
    assert group_status("ANY", statuses) == expected


def test_empty_group_is_missing():
    assert group_status("ALL", []) == "missing"
    assert group_status("ANY", []) == "missing"


def test_unknown_logic_raises():
    with pytest.raises(ValueError):
        group_status("SOME", ["met"])


def test_all_met_approves():
    d = decide([group("1", "ALL", ["1a"], "met"), group("2", "ANY", ["2a", "2b"], "met")],
               [result("1a", "met"), result("2a", "met"), result("2b", "not_met")], True)
    assert d.outcome == "APPROVE" and d.missing_items == []


def test_low_confidence_goes_to_clinician():
    d = decide([group("1", "ALL", ["1a"], "met")], [result("1a", "met", "low")], True)
    assert d.outcome == "SEND_TO_CLINICIAN"
    assert "1a" in d.reasons[0]


def test_unsupported_diagnosis_goes_to_clinician():
    d = decide([group("1", "ALL", ["1a"], "met")], [result("1a", "met")], False)
    assert d.outcome == "SEND_TO_CLINICIAN"
    assert "diagnosis" in d.reasons[0].lower()


def test_group_not_met_goes_to_clinician_never_deny():
    d = decide([group("1", "ALL", ["1a"], "met"), group("2", "ANY", ["2a", "2b"], "not_met")],
               [result("1a", "met"), result("2a", "not_met"), result("2b", "not_met")], True)
    assert d.outcome == "SEND_TO_CLINICIAN"
    assert "Group 2" in d.reasons[0]


def test_not_met_beats_missing():
    d = decide([group("1", "ALL", ["1a"], "missing"), group("2", "ALL", ["2a"], "not_met")],
               [result("1a", "missing"), result("2a", "not_met")], True)
    assert d.outcome == "SEND_TO_CLINICIAN"


def test_missing_requests_info_and_lists_items():
    d = decide([group("1", "ANY", ["1a", "1b"], "missing"), group("2", "ALL", ["2a"], "met")],
               [result("1a", "missing"), result("1b", "not_met"), result("2a", "met")], True)
    assert d.outcome == "REQUEST_INFO"
    assert d.missing_items == ["1a: text 1a"]  # only the missing criterion, not the not_met one


def test_any_group_with_several_missing_says_one_of():
    d = decide([group("1", "ANY", ["1a", "1b"], "missing")], [result("1a", "missing"), result("1b", "missing")], True)
    assert d.missing_items == ["One of: 1a: text 1a", "One of: 1b: text 1b"]


def test_fallback_decision():
    d = fallback_decision("LLM timed out")
    assert d.outcome == "SEND_TO_CLINICIAN" and "Automatic check failed" in d.reasons[0]


def test_unsure_alternative_is_ignored_when_any_group_already_met():
    # Sleep apnea (1a) clearly met; the AI is unsure about parasomnia (1c), which isn't needed.
    d = decide([group("1", "ANY", ["1a", "1c"], "met"), group("2", "ALL", ["2a"], "met")],
               [result("1a", "met"), result("1c", "missing", "low"), result("2a", "met")], True)
    assert d.outcome == "APPROVE"


def test_unsure_answer_still_counts_when_it_could_change_the_outcome():
    # The only met option in the ANY group is itself unsure, so a human must look.
    d = decide([group("1", "ANY", ["1a", "1c"], "met")], [result("1a", "met", "low"), result("1c", "not_met")], True)
    assert d.outcome == "SEND_TO_CLINICIAN"
    # In an ALL group every criterion matters.
    d = decide([group("1", "ALL", ["1a", "1b"], "met")], [result("1a", "met"), result("1b", "met", "low")], True)
    assert d.outcome == "SEND_TO_CLINICIAN"


def test_unsure_missing_asks_for_info_instead_of_clinician():
    d = decide([group("1", "ALL", ["1a"], "met"), group("2", "ANY", ["2a", "2b"], "missing")],
               [result("1a", "met"), result("2a", "missing", "low"), result("2b", "missing")], True)
    assert d.outcome == "REQUEST_INFO"


def test_flagged_missing_still_goes_to_clinician():
    flagged = result("2a", "missing", "low").model_copy(update={"flags": ["Quote not found in notes"]})
    d = decide([group("2", "ALL", ["2a"], "missing")], [flagged], True)
    assert d.outcome == "SEND_TO_CLINICIAN"
