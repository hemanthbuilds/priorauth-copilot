"""Which policy PDF covers each procedure, and which diagnosis codes Medicare accepts for it.

Read from data/procedures.yaml (written by a person) plus the accepted-code lists downloaded by fetch_cms.
It is used to search only the right policy and to check diagnosis codes exactly (not by the LLM).

Check the setup: uv run python -m app.policy_store
"""

import json
import sys
from functools import lru_cache

import yaml

from app import config
from app.models import PolicyInfo


class PolicyError(Exception):
    """procedures.yaml or a file it points to is missing or broken."""


@lru_cache(maxsize=1)
def load() -> dict[str, dict]:
    """{cpt: {name, policy_key, title, policy_id, files, diagnoses}}, checked so a broken setup fails loudly."""
    data = yaml.safe_load(config.PROCEDURES_YAML.read_text())
    table, problems = {}, []
    for p in data["procedures"]:
        missing = [f for f in p["files"] if not (config.POLICY_DIR / f).exists()]
        if missing:
            problems.append(f"CPT {p['cpt']}: policy file(s) not found in {config.POLICY_DIR}: {missing}")
        if p.get("diagnoses_file"):
            path = config.POLICY_DIR / p["diagnoses_file"]
            if not path.exists():
                problems.append(f"CPT {p['cpt']}: {path.name} not found (run app.ingest.fetch_cms)")
                continue
            diagnoses = json.loads(path.read_text())
        else:
            diagnoses = {str(k): v for k, v in (p.get("diagnoses") or {}).items()}
        table[str(p["cpt"])] = {"cpt": str(p["cpt"]), "name": p["name"], "policy_key": p["policy_key"],
                                "title": p["title"], "policy_id": str(p["policy_id"]), "files": p["files"],
                                "diagnoses": diagnoses}
    if problems:
        raise PolicyError("Invalid policy setup:\n- " + "\n- ".join(problems))
    return table


def load_procedures() -> list[dict]:
    return list(load().values())


def procedures() -> dict[str, str]:
    """{cpt: procedure name}, for the UI."""
    return {cpt: row["name"] for cpt, row in load().items()}


def get_policy(cpt: str | None) -> PolicyInfo | None:
    row = load().get((cpt or "").strip().removeprefix("CPT:"))
    if not row:
        return None
    return PolicyInfo(policy_key=row["policy_key"], title=row["title"], policy_id=row["policy_id"],
                      source=", ".join(row["files"]))


def code_check(icd_codes: list[str], cpt: str | None) -> str | None:
    """A plain fact for the LLM and the user: which of the codes the policy's billing article accepts."""
    row = load().get((cpt or "").strip())
    if not row or not icd_codes:
        return None
    codes = [c.strip().upper() for c in icd_codes]
    ok = [c for c in codes if c in row["diagnoses"]]
    bad = [c for c in codes if c not in row["diagnoses"]]
    parts = []
    if ok:
        parts.append(f"{', '.join(ok)} {'is' if len(ok) == 1 else 'are'} on the accepted list for CPT {row['cpt']}.")
    if bad:
        parts.append(f"{', '.join(bad)} {'is' if len(bad) == 1 else 'are'} NOT on the accepted list for CPT "
                     f"{row['cpt']} in the billing article.")
    return " ".join(parts)


def main() -> None:
    try:
        table = load()
    except PolicyError as e:
        print(f"POLICY SETUP INVALID: {e}", file=sys.stderr)
        sys.exit(1)
    for cpt, row in table.items():
        print(f"CPT {cpt}: {row['title']} | files {row['files']} | {len(row['diagnoses']):,} accepted codes")


if __name__ == "__main__":
    main()
