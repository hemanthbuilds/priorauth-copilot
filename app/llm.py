"""The only place that talks to the LLM (Groq). Adds timeouts and retries, and raises LLMError on failure.

Other modules call structured() or text(). Tests replace these two functions with fakes.
"""

import re
import time
from functools import lru_cache

from pydantic import BaseModel, ValidationError

from app import config


class LLMError(Exception):
    """The LLM could not give a usable answer after retries."""


@lru_cache(maxsize=1)
def get_llm():
    if not config.GROQ_API_KEY:
        raise LLMError("GROQ_API_KEY is not set")
    from langchain_groq import ChatGroq

    # max_retries covers network errors and rate limits (429) inside the Groq SDK.
    # gpt-oss models "think" before answering; low effort is plenty for checking one rule and uses far fewer
    # tokens (the free Groq tier allows 200,000 tokens per day).
    extra = {"reasoning_effort": "low"} if config.GROQ_MODEL.startswith("openai/gpt-oss") else {}
    return ChatGroq(model=config.GROQ_MODEL, temperature=0, max_retries=3, timeout=60, **extra)


def _wait(error: Exception, attempt: int) -> None:
    """Pause before retrying. Rate-limit errors say how long to wait ("try again in 13.7s"), so honor that."""
    found = re.search(r"try again in ([\d.]+)s", str(error))
    if found or "rate_limit" in str(error):
        time.sleep(min(float(found.group(1)) + 1 if found else 20, 60))
    else:
        time.sleep(1.5 * (attempt + 1))


def structured(prompt: str, schema: type[BaseModel], retries: int = 2) -> BaseModel:
    """Ask the LLM for an answer that fits a Pydantic model. Retries on invalid output."""
    # gpt-oss models often call tools that don't exist in tool-calling mode (0 of 4 worked in a test on
    # gpt-oss-20b), while Groq's JSON-schema mode forces the exact shape (4 of 4).
    method = "json_schema" if config.GROQ_MODEL.startswith("openai/gpt-oss") else "function_calling"
    llm = get_llm().with_structured_output(schema, method=method)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            result = llm.invoke(prompt)
            if isinstance(result, schema):
                return result
            return schema.model_validate(result)  # some providers return a dict
        except ValidationError as e:
            last_error = e
        except Exception as e:  # tool-call parse errors, timeouts, API errors
            last_error = e
        _wait(last_error, attempt)
    raise LLMError(f"{schema.__name__} failed after {retries + 1} tries: {last_error}")


def text(prompt: str, retries: int = 2) -> str:
    """Ask the LLM for plain text."""
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return str(get_llm().invoke(prompt).content).strip()
        except Exception as e:
            last_error = e
            _wait(e, attempt)
    raise LLMError(f"text call failed after {retries + 1} tries: {last_error}")
