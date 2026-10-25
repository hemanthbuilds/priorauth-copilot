"""Small helpers shared by every Streamlit page: calling the API and showing the safety banner."""

import os
import sys
from pathlib import Path

import httpx
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `app.config` imports from any page
from app import config  # noqa: E402

API = os.getenv("API_URL", config.API_URL)
BANNER = "Decision support only. A human makes the final decision."


def api_call(method: str, path: str, **kwargs) -> tuple[dict | list | None, str | None]:
    """Call the backend. Returns (data, None) on success or (None, a plain-English error)."""
    try:
        r = httpx.request(method, f"{API}{path}", timeout=300, **kwargs)
    except httpx.HTTPError:
        return None, f"I can't reach the checking service at {API}. Is the API running?"
    if r.status_code >= 400:
        detail = r.json().get("detail") if r.headers.get("content-type", "").startswith("application/json") else r.text
        return None, str(detail)
    return r.json(), None


def api(method: str, path: str, **kwargs):
    """Call the backend. Shows the error in the page and returns None on failure."""
    data, error = api_call(method, path, **kwargs)
    if error:
        st.error(error)
    return data


def banner() -> None:
    st.info(f"⚠️ {BANNER}")


def score_color(score: float | None) -> str:
    if score is None:
        return "#9aa0a6"
    return "#1e8e3e" if score >= 0.8 else "#e37400" if score >= 0.6 else "#d93025"


def badge(label: str, score: float | None, pending: bool = False) -> str:
    text = "Scoring…" if pending else ("n/a" if score is None else f"{score:.2f}")
    color = "#9aa0a6" if pending else score_color(score)
    return (f"<span style='background:{color};color:white;padding:2px 8px;border-radius:10px;"
            f"font-size:0.85em;margin-right:6px'>{label}: {text}</span>")
