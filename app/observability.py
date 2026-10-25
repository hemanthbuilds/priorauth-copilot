"""LangSmith tracing setup and small helpers (run URLs, feedback). Everything here is optional:
with no LANGSMITH_API_KEY, tracing is switched off and every helper quietly does nothing.
"""

import logging
import os
from functools import lru_cache

from app import config

log = logging.getLogger("observability")


def _setup() -> bool:
    wants = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    has_key = bool(os.getenv("LANGSMITH_API_KEY", "").strip())
    if wants and not has_key:
        print("WARNING: LANGSMITH_API_KEY is not set, so LangSmith tracing is off. The app works normally.")
    enabled = wants and has_key
    # Make the choice explicit for langsmith and LangChain (they read these env vars).
    os.environ["LANGSMITH_TRACING"] = "true" if enabled else "false"
    os.environ.setdefault("LANGSMITH_PROJECT", config.LANGSMITH_PROJECT)
    return enabled


TRACING_ENABLED = _setup()


@lru_cache(maxsize=1)
def get_client():
    from langsmith import Client
    return Client()


def run_url(run_id: str | None) -> str | None:
    """Link to a trace in the LangSmith UI, or None if tracing is off or the run isn't uploaded yet."""
    if not (TRACING_ENABLED and run_id):
        return None
    try:
        get_client().flush()
        return get_client().read_run(run_id).url
    except Exception as e:
        log.warning("could not get LangSmith run url for %s: %s", run_id, e)
        return None


def add_feedback(run_id: str | None, key: str, score: float | None, comment: str | None = None) -> None:
    """Attach a score (e.g. faithfulness, human_agreed) to a trace. Never raises."""
    if not (TRACING_ENABLED and run_id) or score is None:
        return
    try:
        get_client().create_feedback(run_id, key=key, score=score, comment=comment)
    except Exception as e:
        log.warning("could not send feedback %s to LangSmith: %s", key, e)


def project_url() -> str | None:
    if not TRACING_ENABLED:
        return None
    try:
        return get_client().read_project(project_name=config.LANGSMITH_PROJECT).url
    except Exception:
        return None
