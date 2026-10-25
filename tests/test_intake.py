"""Reading the procedure and diagnosis codes out of chat messages."""

from app.intake import find_cpt, find_icd

PROCS = {"72148": "MRI lumbar spine without contrast", "27447": "Total knee arthroplasty",
         "95810": "Polysomnography (attended sleep study)"}


def test_icd_codes():
    assert find_icd("knee replacement, m17.11 and M17.12") == ["M17.11", "M17.12"]
    assert find_icd("dx G47.33") == ["G47.33"]
    assert find_icd("CPT 27447 for 8 weeks") == []  # numbers aren't diagnosis codes


def test_cpt_by_code_or_words():
    assert find_cpt("please check 27447", PROCS) == "27447"
    assert find_cpt("knee replacement", PROCS) == "27447"
    assert find_cpt("lumbar MRI for back pain", PROCS) == "72148"
    assert find_cpt("sleep study", PROCS) == "95810"
    assert find_cpt("why is rule 1a missing?", PROCS) is None
