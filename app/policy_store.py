"""Which policy and rules apply to a procedure: a plain lookup table built from two human-reviewed files.

procedures.yaml says which policy covers each CPT code and which ICD-10 codes it accepts; rules.json lists
each policy's rules. No LLM is involved, so the same procedure always gets the same policy and rules.
The files are checked when loaded and a broken setup fails loudly.

Check the files: uv run python -m app.policy_store
"""

import json
import sys
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import ValidationError

from app import config
from app.models import CriteriaFile, CriteriaGroup, PolicyInfo


class PolicyError(Exception):
    """procedures.yaml or rules.json is broken."""


def load_procedures(path: Path | None = None) -> list[dict]:
    """Read procedures.yaml: one entry per procedure, with its policy, files and accepted diagnosis codes."""
    data = yaml.safe_load((path or config.PROCEDURES_YAML).read_text())
    return [{"cpt": str(p["cpt"]), "name": p["name"], "policy_key": p["policy_key"], "title": p["title"],
             "policy_id": str(p["policy_id"]), "files": p["files"], "diagnoses": _diagnoses(p)}
            for p in data["procedures"]]


def _diagnoses(block: dict) -> dict[str, str]:
    """ICD-10 codes listed inline, or (for real policies) in a JSON file downloaded by fetch_cms."""
    if block.get("diagnoses_file"):
        path = config.POLICY_DIR / block["diagnoses_file"]
        if not path.exists():
            raise PolicyError(f"{path} not found. Run: uv run python -m app.ingest.fetch_cms")
        return json.loads(path.read_text())
    return {str(k): v for k, v in (block.get("diagnoses") or {}).items()}


def load_criteria(path: Path | None = None) -> CriteriaFile:
    path = path or config.RULES_FILE
    if not path.exists():
        raise PolicyError(f"{path} not found. Run extract_criteria, review the draft, and save it as rules.json.")
    try:
        return CriteriaFile.model_validate(json.loads(path.read_text()))
    except (ValidationError, json.JSONDecodeError) as e:
        raise PolicyError(f"{path} is not valid criteria JSON:\n{e}") from e


def build_table(procedures: list[dict], criteria: CriteriaFile) -> dict[str, dict]:
    """{cpt: procedure + its rule groups}, after checking every procedure has rules and no rule is broken."""
    by_key = {p.policy_key: p for p in criteria.policies}
    problems = [f"rules.json has a policy that is not in procedures.yaml: {k}"
                for k in sorted(set(by_key) - {p["policy_key"] for p in procedures})]
    table = {}
    for proc in procedures:
        policy = by_key.get(proc["policy_key"])
        if not policy or not policy.groups:
            problems.append(f"{proc['policy_key']} (CPT {proc['cpt']}) has no rule groups in rules.json")
            continue
        ids = [c.id for g in policy.groups for c in g.criteria]
        problems += [f"{proc['policy_key']} group {g.group_id} has no criteria" for g in policy.groups if not g.criteria]
        problems += [f"duplicate criterion id {i} in {proc['policy_key']}" for i in sorted(set(ids)) if ids.count(i) > 1]
        table[proc["cpt"]] = {**proc, "groups": policy.groups}
    if problems:
        raise PolicyError("Invalid policy setup:\n- " + "\n- ".join(problems))
    return table


@lru_cache(maxsize=1)
def load() -> dict[str, dict]:
    return build_table(load_procedures(), load_criteria())


def _row(cpt: str) -> dict | None:
    return load().get(cpt.strip().removeprefix("CPT:"))


def supported_cpts() -> list[str]:
    return list(load())


def procedures() -> dict[str, str]:
    """{cpt: procedure name}, for the UI."""
    return {cpt: row["name"] for cpt, row in load().items()}


def get_policy(cpt: str) -> PolicyInfo | None:
    row = _row(cpt)
    if not row:
        return None
    return PolicyInfo(policy_key=row["policy_key"], title=row["title"], policy_id=row["policy_id"],
                      source=", ".join(row["files"]))


def get_criteria(cpt: str) -> list[CriteriaGroup]:
    """The procedure's rule groups, in policy order, with their ALL/ANY logic."""
    row = _row(cpt)
    return list(row["groups"]) if row else []


def diagnosis_supports(icd_codes: list[str], cpt: str) -> bool:
    """True if at least one of the case's ICD-10 codes is on the policy's accepted list."""
    row = _row(cpt)
    return bool(row) and any(c.strip().upper() in row["diagnoses"] for c in icd_codes)


def explain(cpt: str) -> str:
    """Plain-English summary of the policy and its rules (used in the chat prompt)."""
    row = _row(cpt)
    if not row:
        return f"CPT {cpt} is not supported."
    lines = [f"CPT {row['cpt']} ({row['name']}) is covered by {row['title']}."]
    for g in row["groups"]:
        word = "all of" if g.logic == "ALL" else "one of"
        lines.append(f"Group {g.group_id} ({g.description}) needs {word}: "
                     + "; ".join(f"{c.id} (page {c.page})" for c in g.criteria))
    return "\n".join(lines)


def main() -> None:
    try:
        table = load()
    except PolicyError as e:
        print(f"POLICY SETUP INVALID: {e}", file=sys.stderr)
        sys.exit(1)
    for cpt, row in table.items():
        n = sum(len(g.criteria) for g in row["groups"])
        print(f"CPT {cpt}: {row['title']} | {len(row['groups'])} groups, {n} rules, {len(row['diagnoses']):,} codes")


if __name__ == "__main__":
    main()
