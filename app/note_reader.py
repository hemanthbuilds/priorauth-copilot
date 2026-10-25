"""Reads a doctor's note uploaded as a .txt or .pdf file and returns its plain text."""

import io

from pypdf import PdfReader


class NoteReadError(Exception):
    """The file can't be turned into text."""


def read_note(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".txt"):
        text = data.decode("utf-8", errors="replace")
    elif name.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            raise NoteReadError(f"Could not read the PDF: {e}") from e
        if not text.strip():
            # shortcut: no OCR, so scanned (image-only) PDFs can't be read; add OCR if scanned notes are common.
            raise NoteReadError("This PDF has no text layer (it's probably a scanned image). "
                                "Copy the text into the box instead, or use a text-based PDF.")
    else:
        raise NoteReadError("Please upload a .txt or .pdf file.")
    return text.strip()
