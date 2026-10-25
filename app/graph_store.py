"""Reads the knowledge graph and answers exact questions about it: which policy covers a procedure,
what its rules are, and whether a diagnosis supports it. No LLM is involved here.
"""

import json
from functools import lru_cache

import networkx as nx

from app import config
from app.models import CriteriaGroup, Criterion, PolicyInfo


@lru_cache(maxsize=1)
def load_graph() -> nx.DiGraph:
    if not config.GRAPH_FILE.exists():
        raise FileNotFoundError(f"{config.GRAPH_FILE} missing. Run: uv run python -m app.ingest.build_graph")
    return nx.node_link_graph(json.loads(config.GRAPH_FILE.read_text()), edges="edges")


def cpt_node(cpt: str) -> str:
    return cpt if cpt.startswith("CPT:") else f"CPT:{cpt.strip()}"


def _out(g: nx.DiGraph, node: str, edge_type: str) -> list[str]:
    return [v for _, v, d in g.out_edges(node, data=True) if d["type"] == edge_type]


def supported_cpts(g: nx.DiGraph | None = None) -> list[str]:
    g = g or load_graph()
    return [n.removeprefix("CPT:") for n, d in g.nodes(data=True) if d["type"] == "Procedure"]


def get_policy(cpt: str, g: nx.DiGraph | None = None) -> str | None:
    """Graph walk: Procedure -COVERED_BY-> Policy. Returns the policy node id, or None."""
    g = g or load_graph()
    node = cpt_node(cpt)
    if node not in g:
        return None
    policies = _out(g, node, "COVERED_BY")
    return policies[0] if policies else None


def get_policy_info(policy: str, g: nx.DiGraph | None = None) -> PolicyInfo:
    g = g or load_graph()
    d = g.nodes[policy]
    return PolicyInfo(node_id=policy, policy_key=d["policy_key"], title=d["title"],
                      policy_id=d["policy_id"], source=d["source"])


def get_criteria(policy: str, g: nx.DiGraph | None = None) -> list[CriteriaGroup]:
    """All criteria groups of a policy, in policy order, with their logic and criteria."""
    g = g or load_graph()
    groups = []
    for gid in sorted(_out(g, policy, "REQUIRES"), key=lambda n: g.nodes[n]["order"]):
        gd = g.nodes[gid]
        crits = sorted(_out(g, gid, "HAS_OPTION"), key=lambda n: g.nodes[n]["order"])
        groups.append(CriteriaGroup(
            group_id=gid.rsplit(":", 1)[1], logic=gd["logic"], description=gd["description"],
            criteria=[Criterion(id=g.nodes[c]["short_id"], text=g.nodes[c]["text"], page=g.nodes[c]["page"])
                      for c in crits]))
    return groups


def diagnosis_supports(icd_codes: list[str], cpt: str, g: nx.DiGraph | None = None) -> bool:
    """True if at least one of the case's ICD-10 codes has a SUPPORTS edge to the procedure."""
    g = g or load_graph()
    node = cpt_node(cpt)
    return any(g.has_edge(f"ICD:{c.strip().upper()}", node) for c in icd_codes)


def graph_path(cpt: str, g: nx.DiGraph | None = None) -> list[str]:
    """Node list CPT → policy → groups → criteria, for the UI and the trace."""
    g = g or load_graph()
    policy = get_policy(cpt, g)
    if not policy:
        return []
    path = [cpt_node(cpt), policy]
    for gid in sorted(_out(g, policy, "REQUIRES"), key=lambda n: g.nodes[n]["order"]):
        path.append(gid)
        path.extend(sorted(_out(g, gid, "HAS_OPTION"), key=lambda n: g.nodes[n]["order"]))
    return path


def explain_path(cpt: str, g: nx.DiGraph | None = None) -> str:
    """Plain-English version of the path, e.g. for the chat prompt."""
    g = g or load_graph()
    policy = get_policy(cpt, g)
    if not policy:
        return f"CPT {cpt} is not in the knowledge graph."
    lines = [f"CPT {cpt} ({g.nodes[cpt_node(cpt)]['name']}) is covered by {g.nodes[policy]['title']}."]
    for grp in get_criteria(policy, g):
        word = "all of" if grp.logic == "ALL" else "one of"
        lines.append(f"Group {grp.group_id} ({grp.description}) needs {word}: "
                     + "; ".join(f"{c.id} (page {c.page})" for c in grp.criteria))
    return "\n".join(lines)


def graph_data(g: nx.DiGraph | None = None) -> dict:
    """Nodes and edges as plain JSON (for the API and the pyvis view)."""
    return nx.node_link_data(g or load_graph(), edges="edges")
