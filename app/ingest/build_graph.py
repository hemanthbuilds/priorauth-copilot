"""Builds the knowledge graph from procedures.yaml (hand-written) + criteria.json (human-reviewed).

Graph: Procedure -COVERED_BY-> Policy -REQUIRES-> CriteriaGroup -HAS_OPTION-> Criterion,
       Diagnosis -SUPPORTS-> Procedure.
The graph is validated and the build fails loudly if anything is broken.

Run: uv run python -m app.ingest.build_graph
"""

import json
import sys
from pathlib import Path

import networkx as nx
import yaml
from pydantic import ValidationError

from app import config
from app.models import CriteriaFile


class GraphError(Exception):
    """The graph or its inputs are invalid."""


def load_procedures(path: Path | None = None) -> list[dict]:
    """Read procedures.yaml and flatten the sample or cms block into each procedure."""
    data = yaml.safe_load((path or config.PROCEDURES_YAML).read_text())
    mode = "sample" if config.USE_SAMPLE_DATA else "cms"
    procedures = []
    for p in data["procedures"]:
        block = p[mode]
        procedures.append({
            "cpt": str(p["cpt"]), "name": p["name"], "policy_key": p["policy_key"],
            "title": block["title"], "policy_id": str(block["policy_id"]), "files": block["files"],
            "diagnoses": _diagnoses(block),
        })
    return procedures


def _diagnoses(block: dict) -> dict[str, str]:
    """ICD-10 codes listed inline, or (for real policies) in a JSON file downloaded by fetch_cms."""
    if block.get("diagnoses_file"):
        path = config.POLICY_DIR / block["diagnoses_file"]
        if not path.exists():
            raise GraphError(f"{path} not found. Run: uv run python -m app.ingest.fetch_cms")
        return json.loads(path.read_text())
    return {str(k): v for k, v in (block.get("diagnoses") or {}).items()}


def load_criteria(path: Path | None = None) -> CriteriaFile:
    path = path or config.CRITERIA_FILE
    if not path.exists():
        raise GraphError(f"{path} not found. Run extract_criteria, review the draft, and save it as criteria.json.")
    try:
        return CriteriaFile.model_validate(json.loads(path.read_text()))
    except (ValidationError, json.JSONDecodeError) as e:
        raise GraphError(f"{path} is not valid criteria JSON:\n{e}") from e


def build_graph(procedures: list[dict], criteria: CriteriaFile) -> nx.DiGraph:
    g = nx.DiGraph()
    by_key = {p.policy_key: p for p in criteria.policies}
    known_keys = {p["policy_key"] for p in procedures}
    unknown = set(by_key) - known_keys
    if unknown:
        raise GraphError(f"criteria.json has policies not in procedures.yaml: {sorted(unknown)}")

    for proc in procedures:
        cpt = f"CPT:{proc['cpt']}"
        pol = f"POLICY:{proc['policy_key']}"
        g.add_node(cpt, type="Procedure", name=proc["name"])
        g.add_node(pol, type="Policy", title=proc["title"], policy_id=proc["policy_id"],
                   source=", ".join(proc["files"]), policy_key=proc["policy_key"])
        g.add_edge(cpt, pol, type="COVERED_BY")
        for code, name in proc["diagnoses"].items():
            g.add_node(f"ICD:{code}", type="Diagnosis", name=name)
            g.add_edge(f"ICD:{code}", cpt, type="SUPPORTS")

        policy = by_key.get(proc["policy_key"])
        for order, grp in enumerate(policy.groups if policy else []):
            gid = f"GROUP:{proc['policy_key']}:{grp.group_id}"
            g.add_node(gid, type="CriteriaGroup", logic=grp.logic, description=grp.description, order=order)
            g.add_edge(pol, gid, type="REQUIRES")
            for c_order, crit in enumerate(grp.criteria):
                cid = f"CRIT:{proc['policy_key']}:{crit.id}"
                if cid in g:
                    raise GraphError(f"duplicate criterion id {crit.id} in policy {proc['policy_key']}")
                g.add_node(cid, type="Criterion", text=crit.text, page=crit.page, short_id=crit.id, order=c_order)
                g.add_edge(gid, cid, type="HAS_OPTION")
    validate_graph(g)
    return g


def validate_graph(g: nx.DiGraph) -> None:
    """Every procedure has exactly one policy, every policy ≥1 group, every group ≥1 criterion."""
    problems = []

    def out(node, edge_type):
        return [v for _, v, d in g.out_edges(node, data=True) if d["type"] == edge_type]

    for n, d in g.nodes(data=True):
        if d["type"] == "Procedure" and len(out(n, "COVERED_BY")) != 1:
            problems.append(f"{n} must have exactly one policy, has {len(out(n, 'COVERED_BY'))}")
        if d["type"] == "Policy" and not out(n, "REQUIRES"):
            problems.append(f"{n} has no criteria groups (missing from criteria.json?)")
        if d["type"] == "CriteriaGroup" and not out(n, "HAS_OPTION"):
            problems.append(f"{n} has no criteria")
    if problems:
        raise GraphError("Invalid knowledge graph:\n- " + "\n- ".join(problems))


def save_graph(g: nx.DiGraph, path: Path | None = None) -> None:
    path = path or config.GRAPH_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nx.node_link_data(g, edges="edges"), indent=2))


def main() -> None:
    try:
        g = build_graph(load_procedures(), load_criteria())
    except GraphError as e:
        print(f"GRAPH BUILD FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    save_graph(g)
    counts = {}
    for _, d in g.nodes(data=True):
        counts[d["type"]] = counts.get(d["type"], 0) + 1
    print(f"Graph OK → {config.GRAPH_FILE}  nodes={counts}  edges={g.number_of_edges()}")


if __name__ == "__main__":
    main()
