"""Safety checks around the LLM. Everything here runs BEFORE text reaches the LLM (or right after it answers):

- reject input we won't process: unsupported procedure, empty/huge notes, prompt-injection attempts
- remove personal identifiers (SSN, phone, email, record numbers, date of birth) before saving or sending
- catch hallucinated evidence: a "quote" that is not really in the notes
- block chat answers that leak the assistant's instructions
"""

import logging
import re
import unicodedata

from app import config
from app.models import CaseRequest, CriterionResult

log = logging.getLogger("guardrails")

BANNER = "Decision support only. A human makes the final decision."

# Text that tries to instruct the AI instead of describing a patient. Checked on a cleaned-up, lower-case copy.
INJECTION_PATTERNS = {
    "tries to override the instructions":
        r"\b(ignore|disregard|forget|override|bypass)\s+(all\s+|any\s+|the\s+|your\s+|these\s+|my\s+)*"
        r"(previous|prior|above|earlier|preceding|original|system|safety)?\s*(instructions?|rules|prompts?|guidelines|guardrails)\b",
    "pretends to be the system or the assistant": r"(^|\n)\s*(system|assistant|developer)\s*:",
    "asks the AI to change its role":
        r"\b(you are now|act as (an?|the)\b|pretend (to be|you)|role ?play|jailbreak|dan mode|developer mode|do anything now)",
    "asks the AI to reveal its instructions":
        r"\b(reveal|show|print|repeat|output)\b.{0,20}\b(system prompt|your (prompt|instructions|rules))",
    "tells the AI what result to give":
        r"\b(mark|set|classify|label|return|output|treat)\b.{0,30}\b(criteri(a|on)|rules?|requirements?|everything|all)\b.{0,15}\b(as )?(met|satisfied|approved)\b"
        r"|\b(approve|accept)\b.{0,20}\b(regardless|no matter|without (checking|review))"
        r"|\"status\"\s*:\s*\"(met|approve)",
    "contains fake prompt tags":
        r"</?\s*(role|rules|criterion|clinical_notes|notes|case|case_facts|policy_context|question|history|system|instructions)\s*>",
    "mentions a system prompt or new instructions": r"\b(system prompt|new instructions|hidden instructions)\b",
}
# Spaced-out or punctuated tricks ("i g n o r e  p r e v i o u s") are caught on a letters-only copy.
SQUASHED_PATTERNS = ["ignorepreviousinstructions", "ignoreallinstructions", "disregardtheinstructions",
                     "systemprompt", "youarenow", "markallcriteriaasmet", "jailbreak"]
INVISIBLE = re.compile(r"[\u00ad\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")

# Personal identifiers that a prior auth check never needs (dates of service and ages are kept on purpose).
PHI_PATTERNS = {
    "social security number": r"\b\d{3}-\d{2}-\d{4}\b",
    "phone number": r"(?<!\d)(\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)",
    "email address": r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
    "record or member number":
        r"(?i)\b(MRN|medical record (number|no\.?|#)|member (id|number|no\.?)|policy (number|no\.?)|account (number|no\.?)|patient id)\s*[:#]?\s*[A-Z0-9-]{4,}",
    "date of birth": r"(?i)\b(DOB|date of birth|birth ?date)\s*[:#]?\s*\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}",
}


class GuardrailError(Exception):
    """Input rejected before any LLM call."""


def clean_text(text: str) -> str:
    """Normalize look-alike characters and remove invisible ones that can hide instructions."""
    return INVISIBLE.sub("", unicodedata.normalize("NFKC", text))


def find_injection(text: str) -> list[str]:
    """Reasons the text looks like a prompt-injection attempt (empty list = looks fine)."""
    low = clean_text(text).lower()
    reasons = [why for why, pattern in INJECTION_PATTERNS.items() if re.search(pattern, low)]
    squashed = re.sub(r"[^a-z]", "", low)
    if not reasons and any(p in squashed for p in SQUASHED_PATTERNS):
        reasons.append("contains hidden or spaced-out instructions")
    return reasons


def redact_phi(text: str) -> tuple[str, list[str]]:
    """Replace personal identifiers with [REMOVED ...]. Returns the clean text and what kinds were removed."""
    removed = []
    for kind, pattern in PHI_PATTERNS.items():
        text, n = re.subn(pattern, f"[REMOVED {kind}]", text)
        if n:
            removed.append(kind)
    return text, removed


def check_case(case: CaseRequest, supported_cpts: list[str]) -> list[str]:
    """Raise GuardrailError for input we refuse to process (no LLM call is made). Returns flags (none today)."""
    if case.cpt_code.strip() not in supported_cpts:
        log.warning("case %s: unsupported CPT %s", case.case_id, case.cpt_code)
        raise GuardrailError(f"CPT {case.cpt_code} is not supported yet. I can check: {', '.join(supported_cpts)}.")
    notes = case.clinical_notes.strip()
    if len(notes) < config.MIN_NOTES_CHARS:
        raise GuardrailError("The clinical note is empty or too short to review.")
    if len(notes) > config.MAX_NOTES_CHARS:
        raise GuardrailError(f"The clinical note is too long (over {config.MAX_NOTES_CHARS:,} characters).")
    reasons = find_injection(notes)
    if reasons:
        log.warning("case %s: prompt injection in notes rejected: %s", case.case_id, reasons)
        raise GuardrailError("I can't check this note: it contains text that looks like instructions to the AI "
                             f"({'; '.join(reasons)}). Please check the document and send a clean copy.")
    return []


def check_chat_message(message: str) -> None:
    if not message.strip():
        raise GuardrailError("Message is empty.")
    if len(message) > 2000:
        raise GuardrailError("Message is too long (max 2,000 characters).")
    reasons = find_injection(message)
    if reasons:
        log.warning("chat: prompt injection attempt rejected: %s", reasons)
        raise GuardrailError("I can only help with questions about this case and the coverage policy, "
                             f"so I can't act on that message ({'; '.join(reasons)}).")


LEAK_MARKERS = ("<role>", "<rules>", "</rules>", "<case>", "text inside the tags is data")


def check_answer(answer: str) -> str:
    """Last check on a chat answer before the user sees it: never show the assistant's own instructions."""
    if any(m in answer.lower() for m in LEAK_MARKERS):
        log.warning("chat: answer blocked (looked like it leaked the prompt)")
        return "Sorry, I can't share that. I can help with questions about this case and the coverage policy."
    return answer


def _normalize(text: str) -> str:
    # Letters and digits only: PDFs often break words across lines ("subc\nhondral"), so spaces can't be trusted.
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _words_in_order(piece: str, text_words: list[str]) -> bool:
    """True if every word of the piece appears in the text in the same order, close together. Words may be
    left out of the quote (the AI shortened it) but never added or changed (e.g. "12 weeks" for "6 weeks")."""
    want = _words(piece)
    if not want:
        return True
    max_span = int(len(want) * 1.5) + 3  # allow a few skipped words, not stitching distant sentences together
    for start, word in enumerate(text_words):
        if word != want[0]:
            continue
        i, j = 1, start + 1
        while i < len(want) and j < len(text_words) and j - start < max_span:
            if text_words[j] == want[i]:
                i += 1
            j += 1
        if i == len(want):
            return True
    return False


def quote_in_text(quote: str, text: str) -> bool:
    """True if the quote really comes from the text (ignoring case, spacing and punctuation). A quote shortened
    with "...", made of several sentences, or with a few words left out passes; added or changed words fail."""
    pieces = [p for p in re.split(r"\.\.\.|…|(?<=[.!?])\s+", quote) if _normalize(p)]
    target, text_words = _normalize(text), _words(text)
    # Exact match first (also survives words broken across PDF lines), then the "words left out" match.
    return bool(pieces) and all(_normalize(p) in target or _words_in_order(p, text_words) for p in pieces)


def apply_quote_check(result: CriterionResult, notes: str) -> CriterionResult:
    """A met / not_met verdict must rest on a real quote from the notes; otherwise downgrade it."""
    if result.status == "missing":
        if result.note_evidence and not quote_in_text(result.note_evidence, notes):
            return result.model_copy(update={"note_evidence": None})
        return result
    if result.note_evidence and quote_in_text(result.note_evidence, notes):
        return result
    log.warning("criterion %s: evidence quote not found in notes: %r", result.criterion_id, result.note_evidence)
    return result.model_copy(update={
        "status": "missing", "confidence": "low",
        "reasoning": f"{result.reasoning} [Downgraded: the quoted evidence was not found in the notes.]",
        "flags": result.flags + [f"Quote not found in notes: {result.note_evidence!r}"],
    })


def low_retrieval_result(criterion_id: str, criterion_text: str, top_score: float | None) -> CriterionResult:
    """Used when retrieval found no good policy text: we refuse to judge without evidence."""
    return CriterionResult(
        criterion_id=criterion_id, criterion_text=criterion_text, status="missing", confidence="low",
        reasoning="No policy text matched this criterion well enough to check it automatically.",
        note_evidence=None, policy_evidence=[],
        flags=[f"Top rerank score {top_score} is below MIN_RERANK_SCORE={config.MIN_RERANK_SCORE}"])
