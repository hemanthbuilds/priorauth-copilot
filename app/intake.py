"""Pulls the procedure (CPT) and diagnosis codes (ICD-10) out of a chat message, e.g. "knee replacement, M17.11"."""

import re

ICD10 = re.compile(r"\b([A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?)\b")
STOP = {"with", "without", "and", "the", "study", "attended", "contrast"}


def find_icd(text: str) -> list[str]:
    """ICD-10 codes like M54.16 or G47.33 (letter, digit, digit/letter, optional .xxxx), upper-cased, in order."""
    found = []
    for code in ICD10.findall(text.upper()):
        if "." in code or len(code) == 3:
            if code not in found:
                found.append(code)
    return found


def find_cpt(text: str, procedures: dict[str, str]) -> str | None:
    """A supported CPT code typed directly, or a word from a procedure's name ("knee", "mri", "sleep")."""
    for cpt in procedures:
        if re.search(rf"\b{cpt}\b", text):
            return cpt
    words = set(re.findall(r"[a-z]{3,}", text.lower()))
    for cpt, name in procedures.items():
        name_words = set(re.findall(r"[a-z]{3,}", name.lower())) - STOP
        if words & name_words:
            return cpt
    return None
