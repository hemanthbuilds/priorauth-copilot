"""Downloads the real Medicare coverage policies (LCDs) and their covered ICD-10 codes from the public
CMS Coverage API (https://api.coverage.cms.gov) into data/policies/.

For each policy it writes:
  data/policies/<LCD>.pdf         the policy as a simple, readable PDF (coverage indications, then the
                                  documentation / associated information section); the app reads this file
  data/policies/<LCD>_icd10.json  the ICD-10 codes the billing article accepts {code: description}

Which LCD and article to fetch is set in data/procedures.yaml. Run: uv run python -m app.ingest.fetch_cms
"""
import html
import json
import re

import httpx
import yaml
from fpdf import FPDF

from app import config

API = "https://api.coverage.cms.gov/v1"


def html_to_text(raw: str | None) -> str:
    """CMS returns the policy sections as HTML; keep the words, list bullets and paragraph breaks."""
    text = html.unescape(raw or "")  # the API escapes its HTML (&lt;p&gt;), so decode before stripping tags
    text = re.sub(r"(?i)<li[^>]*>", "\n- ", text)
    text = re.sub(r"(?i)<br\s*/?>|</p>|</h\d>|</tr>|</div>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


# The built-in PDF fonts only cover Latin-1, so swap the few typographic characters CMS uses.
PLAIN = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-",
                       "\u2022": "-", "\u2026": "...", "\u2265": ">=", "\u2264": "<="})


def write_pdf(path, title: str, subtitle: str, sections: list[tuple[str, str]]) -> None:
    """A plain, readable PDF: title, source line, then each section as a heading and paragraphs."""
    pdf = FPDF(format="Letter")
    pdf.set_margins(20, 20, 20)
    pdf.set_auto_page_break(True, margin=20)
    pdf.add_page()

    def clean(t: str) -> str:
        return t.translate(PLAIN).encode("latin-1", "replace").decode("latin-1")

    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 8, clean(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, clean(subtitle), new_x="LMARGIN", new_y="NEXT")
    for heading, body in sections:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 7, clean(heading), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for para in body.split("\n"):
            if para.strip():
                pdf.multi_cell(0, 5, clean(para.strip()), new_x="LMARGIN", new_y="NEXT")
                pdf.ln(1)
    pdf.output(str(path))


def latest_version(client: httpx.Client, report: str, id_key: str, doc_id: int) -> int:
    """The API needs a version number; the report lists the current version of every final LCD/article."""
    rows = client.get(f"{API}/reports/{report}", params={"state": "all"}).json()["data"]
    for r in rows:
        if r.get(id_key) == doc_id:
            return r["document_version"]
    raise ValueError(f"{doc_id} not found in {report}")


def main() -> None:
    data = yaml.safe_load(config.PROCEDURES_YAML.read_text())
    out_dir = config.POLICY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60) as client:
        token = client.get(f"{API}/metadata/license-agreement").json()["data"][0]["Token"]
        client.headers["Authorization"] = f"Bearer {token}"
        for p in data["procedures"]:
            lcd_id = int(str(p["policy_id"]).lstrip("L"))
            article_id = int(str(p["article_id"]).lstrip("A"))
            lcd_ver = latest_version(client, "local-coverage-final-lcds", "document_id", lcd_id)
            art_ver = latest_version(client, "local-coverage-articles", "document_id", article_id)

            lcd = client.get(f"{API}/data/lcd", params={"lcdid": lcd_id, "ver": lcd_ver}).json()["data"][0]
            pdf_path = out_dir / f"L{lcd_id}.pdf"
            write_pdf(pdf_path, f"{lcd['title']} (L{lcd_id})",
                      f"Medicare Local Coverage Determination, version {lcd_ver}. Downloaded from the CMS Coverage "
                      f"API (api.coverage.cms.gov). Billing article: A{article_id}.",
                      [("Coverage Indications, Limitations, and/or Medical Necessity", html_to_text(lcd.get("indication"))),
                       ("Associated Information", html_to_text(lcd.get("associated_info")))])

            rows = client.get(f"{API}/data/article/icd10-covered",
                              params={"articleid": article_id, "ver": art_ver}).json()["data"]
            codes = {r["icd10_code_id"]: r.get("description", "") for r in rows}
            (out_dir / f"L{lcd_id}_icd10.json").write_text(json.dumps(codes, indent=1, sort_keys=True))
            print(f"CPT {p['cpt']}: {pdf_path.name} (L{lcd_id} v{lcd_ver}), A{article_id} v{art_ver} "
                  f"({len(codes):,} covered ICD-10 codes)")
    print(f"Saved to {out_dir}. Next: check rules.json against the PDFs, then run chunk_index.")


if __name__ == "__main__":
    main()
