"""Policy lookup: exact answers from the shipped sample files, and loud failure on a broken setup."""

import json
import re
import subprocess
import sys

import pytest

from app import config, policy_store
from app.models import CriteriaFile
from app.policy_store import PolicyError, build_table, load_criteria, load_procedures


def raw_criteria() -> dict:
    return json.loads(config.RULES_FILE.read_text())


def test_policy_lookup_is_exact():
    assert policy_store.get_policy("72148").policy_key == "lumbar_mri"
    assert policy_store.get_policy("CPT:27447").policy_key == "total_knee"
    assert policy_store.get_policy("99999") is None
    assert set(policy_store.supported_cpts()) == {"72148", "27447", "95810"}


def test_rule_groups_and_logic():
    groups = policy_store.get_criteria("72148")
    assert [(g.group_id, g.logic) for g in groups] == [("1", "ANY"), ("2", "ALL")]
    assert [c.id for c in groups[0].criteria] == ["1a", "1b"]


def test_diagnosis_supports():
    assert policy_store.diagnosis_supports(["M54.16"], "72148")
    assert policy_store.diagnosis_supports(["z99.9", "m54.16"], "72148")  # one good code is enough
    assert not policy_store.diagnosis_supports(["M54.50"], "95810")  # back pain code for a sleep study
    assert not policy_store.diagnosis_supports(["M54.16"], "99999")


def test_explain():
    assert "one of" in policy_store.explain("95810") and "not supported" in policy_store.explain("99999")


@pytest.mark.parametrize("break_it, message", [
    (lambda d: d["policies"][0]["groups"][0].update(criteria=[]), "has no criteria"),
    (lambda d: d.update(policies=d["policies"][1:]), "lumbar_mri (CPT 72148) has no rule groups"),
    (lambda d: d["policies"][0].update(policy_key="not_a_policy"), "not in procedures.yaml"),
    (lambda d: d["policies"][0]["groups"][1]["criteria"][0].update(id="1a"), "duplicate"),
])
def test_broken_setup_fails_loudly(break_it, message):
    data = raw_criteria()
    break_it(data)
    with pytest.raises(PolicyError, match=re.escape(message)):
        build_table(load_procedures(), CriteriaFile.model_validate(data))


def test_bad_logic_or_bad_json_fails(tmp_path):
    data = raw_criteria()
    data["policies"][0]["groups"][0]["logic"] = "SOME"
    bad = tmp_path / "rules.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(PolicyError, match="not valid"):
        load_criteria(bad)
    bad.write_text("{not json")
    with pytest.raises(PolicyError, match="not valid"):
        load_criteria(bad)


def test_cli_exits_nonzero_on_broken_criteria(tmp_path):
    data = raw_criteria()
    data["policies"][0]["groups"][0]["criteria"] = []
    bad = tmp_path / "rules.json"
    bad.write_text(json.dumps(data))
    code = (f"from pathlib import Path; from app import config; config.RULES_FILE = Path(r'{bad}'); "
            "import app.policy_store as p; p.main()")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=config.ROOT,
                          env={"DATA_DIR": str(config.DATA), "PATH": ""})
    assert proc.returncode == 1 and "POLICY SETUP INVALID" in proc.stderr
