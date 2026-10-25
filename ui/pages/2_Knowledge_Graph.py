"""Knowledge Graph page: an interactive pyvis view of procedures, policies, rule groups, criteria and
diagnoses. Picking a procedure highlights the exact path the app walks for it."""

import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from common import api, banner

st.set_page_config(page_title="Knowledge graph", page_icon="🕸️", layout="wide")
st.title("Knowledge graph")
banner()

data = api("GET", "/api/graph")
if not data:
    st.stop()
nodes = {n["id"]: n for n in data["nodes"]}
edges = data["edges"]
COLORS = {"Procedure": "#1a73e8", "Policy": "#9334e6", "CriteriaGroup": "#e37400", "Criterion": "#1e8e3e",
          "Diagnosis": "#5f6368"}

procs = sorted(n for n, d in nodes.items() if d["type"] == "Procedure")
choice = st.selectbox("Highlight a procedure's path", ["(none)"] + [f"{p} {nodes[p]['name']}" for p in procs])
n_dx = sum(1 for d in nodes.values() if d["type"] == "Diagnosis")
# Real billing articles list thousands of codes, which would freeze the browser, so they start hidden.
show_dx = st.checkbox(f"Show diagnosis codes ({n_dx})", value=n_dx <= 100)


def path_for(cpt: str) -> set[str]:
    """CPT → policy → groups → criteria, plus the diagnoses that support it (same walk as the backend)."""
    keep = {cpt}
    frontier = [cpt]
    while frontier:
        node = frontier.pop()
        for e in edges:
            if e["source"] == node and e["type"] in ("COVERED_BY", "REQUIRES", "HAS_OPTION"):
                keep.add(e["target"])
                frontier.append(e["target"])
    keep |= {e["source"] for e in edges if e["target"] == cpt and e["type"] == "SUPPORTS"}
    return keep


highlight = path_for(choice.split()[0]) if choice != "(none)" else set()

net = Network(height="680px", width="100%", directed=True, cdn_resources="remote")
for nid, d in nodes.items():
    if d["type"] == "Diagnosis" and not show_dx:
        continue
    dim = bool(highlight) and nid not in highlight
    label = nid if d["type"] != "Criterion" else f"{d['short_id']}"
    title = d.get("text") or d.get("title") or d.get("description") or d.get("name") or nid
    if d["type"] == "CriteriaGroup":
        label = f"{nid.split(':')[-1]} ({d['logic']})"
    net.add_node(nid, label=label, title=title, color="#e0e0e0" if dim else COLORS[d["type"]],
                 size=28 if nid in highlight else 16, shape="dot")
for e in edges:
    if e["source"] in net.get_nodes() and e["target"] in net.get_nodes():
        on = e["source"] in highlight and e["target"] in highlight
        net.add_edge(e["source"], e["target"], title=e["type"], color="#202124" if on else "#c0c0c0",
                     width=3 if on else 1)
net.toggle_physics(True)
components.html(net.generate_html(), height=700)

st.caption(" · ".join(f"<span style='color:{c}'>●</span> {t}" for t, c in COLORS.items()), unsafe_allow_html=True)
