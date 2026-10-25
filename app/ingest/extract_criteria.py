"""Uses the LLM to read each policy and draft its coverage criteria (groups, ALL/ANY logic, page numbers).

The output is only a DRAFT (rules_draft.json). A human must review it and save it as rules.json
before the app uses it. The LLM never decides on its own which rules apply.

Run: uv run python -m app.ingest.extract_criteria
"""

import json
import sys

from app import config, llm
from app.policy_store import load_procedures
from app.ingest.chunk_index import load_policy_pages
from app.models import CriteriaFile, PolicyCriteria
from app.prompts import CRITERIA_EXTRACT_PROMPT

# shortcut: long CMS policies are cut to this many characters to fit Groq's token limits;
# if the indications section is cut off, extract that section by hand or use a larger-context model.
MAX_POLICY_CHARS = 30_000


def extract_policy(proc: dict) -> PolicyCriteria:
    pages = load_policy_pages(proc)
    text = "\n\n".join(f"[Page {d.metadata['page']}] ({d.metadata['source']})\n{d.page_content}" for d in pages)
    prompt = CRITERIA_EXTRACT_PROMPT.format(policy_key=proc["policy_key"], policy_text=text[:MAX_POLICY_CHARS])
    result = llm.structured(prompt, PolicyCriteria)
    result.policy_key = proc["policy_key"]  # never trust the LLM with the key
    return result


def main() -> None:
    if not config.GROQ_API_KEY:
        sys.exit("GROQ_API_KEY is not set: cannot draft criteria. Set it in .env and re-run.")
    policies = []
    for proc in load_procedures():
        print(f"Extracting criteria for {proc['policy_id']} ...")
        policies.append(extract_policy(proc))
    config.RULES_DRAFT_FILE.write_text(json.dumps(CriteriaFile(policies=policies).model_dump(), indent=2))
    print(f"Draft written to {config.RULES_DRAFT_FILE}\n"
          f"REVIEW IT, fix anything wrong, then save it as {config.RULES_FILE} and run: uv run python -m app.policy_store")


if __name__ == "__main__":
    main()
