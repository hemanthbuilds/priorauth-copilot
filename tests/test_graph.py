"""Knowledge graph: built from the shipped sample files, exact lookups, and loud failure on broken input."""

import json
import subprocess
import sys

import pytest

from app import config, graph_store
from app.ingest.build_graph import GraphError, build_graph, load_criteria, load_procedures
from app.models import CriteriaFile


@pytest.fixture(scope="module")
def graph():
    return build_graph(load_procedures(), load_criteria())


def raw_criteria() -> dict:
    return json.loads(config.CRITERIA_FILE.read_text())


def test_policy_lookup_is_exact(graph):
    assert graph_store.get_policy("72148", graph) == "POLICY:lumbar_mri"
    assert graph_store.get_policy("CPT:27447", graph) == "POLICY:total_knee"
    assert graph_store.get_policy("99999", graph) is None


def test_criteria_groups_and_logic(graph):
    groups = graph_store.get_criteria("POLICY:lumbar_mri", graph)
    assert [(g.group_id, g.logic) for g in groups] == [("1", "ANY"), ("2", "ALL")]
    assert [c.id for c in groups[0].criteria] == ["1a", "1b"]


def test_diagnosis_supports(graph):
    assert graph_store.diagnosis_supports(["M54.16"], "72148", graph)
    assert graph_store.diagnosis_supports(["z99.9", "m54.16"], "72148", graph)  # one good code is enough
    assert not graph_store.diagnosis_supports(["M54.50"], "95810", graph)  # back pain code for a sleep study


def test_graph_path(graph):
    path = graph_store.graph_path("95810", graph)
    assert path[:3] == ["CPT:95810", "POLICY:sleep_study", "GROUP:sleep_study:1"]
    assert "CRIT:sleep_study:2b" in path
    assert "one of" in graph_store.explain_path("95810", graph)


def test_empty_group_fails_loudly():
    data = raw_criteria()
    data["policies"][0]["groups"][0]["criteria"] = []
    with pytest.raises(GraphError, match="has no criteria"):
        build_graph(load_procedures(), CriteriaFile.model_validate(data))


def test_policy_without_groups_fails_loudly():
    data = raw_criteria()
    data["policies"] = data["policies"][1:]  # lumbar_mri now has no rules
    with pytest.raises(GraphError, match="POLICY:lumbar_mri has no criteria groups"):
        build_graph(load_procedures(), CriteriaFile.model_validate(data))


def test_unknown_policy_key_fails():
    data = raw_criteria()
    data["policies"][0]["policy_key"] = "not_a_policy"
    with pytest.raises(GraphError, match="not in procedures.yaml"):
        build_graph(load_procedures(), CriteriaFile.model_validate(data))


def test_duplicate_criterion_fails():
    data = raw_criteria()
    data["policies"][0]["groups"][1]["criteria"][0]["id"] = "1a"
    with pytest.raises(GraphError, match="duplicate"):
        build_graph(load_procedures(), CriteriaFile.model_validate(data))


def test_bad_logic_or_bad_json_fails(tmp_path):
    data = raw_criteria()
    data["policies"][0]["groups"][0]["logic"] = "SOME"
    bad = tmp_path / "criteria.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(GraphError, match="not valid"):
        load_criteria(bad)
    bad.write_text("{not json")
    with pytest.raises(GraphError, match="not valid"):
        load_criteria(bad)


def test_cli_exits_nonzero_on_broken_criteria(tmp_path):
    # Run the real command against a broken criteria.json and check it fails loudly.
    data = raw_criteria()
    data["policies"][0]["groups"][0]["criteria"] = []
    bad = tmp_path / "criteria.json"
    bad.write_text(json.dumps(data))
    code = ("import sys; from pathlib import Path; from app import config; "
            f"config.CRITERIA_FILE = Path(r'{bad}'); config.GRAPH_FILE = Path(r'{tmp_path}') / 'g.json'; "
            "import app.ingest.build_graph as b; b.main()")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=config.ROOT)
    assert proc.returncode == 1
    assert "GRAPH BUILD FAILED" in proc.stderr
