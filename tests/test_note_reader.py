"""Reading uploaded doctor's notes: text files, text PDFs, and clear errors for scans and other formats."""

import pytest

from app.note_reader import NoteReadError, read_note


def tiny_pdf(text: str | None) -> bytes:
    """Build a one-page PDF by hand; with text=None the page has no text (like a scanned image)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode() if text else b""
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = b"%PDF-1.4\n", []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return out


def test_reads_text_file():
    assert read_note("note.TXT", "Back pain for 8 weeks.\n".encode()) == "Back pain for 8 weeks."


def test_reads_text_pdf():
    assert "Completed 6 weeks of physical therapy" in read_note("note.pdf", tiny_pdf("Completed 6 weeks of physical therapy"))


def test_scanned_pdf_gives_clear_error():
    with pytest.raises(NoteReadError, match="scanned"):
        read_note("scan.pdf", tiny_pdf(None))


def test_other_formats_rejected():
    with pytest.raises(NoteReadError, match=".txt or .pdf"):
        read_note("note.docx", b"PK...")
