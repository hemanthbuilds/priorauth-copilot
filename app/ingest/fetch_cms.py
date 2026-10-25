"""Downloads the real Medicare coverage policies (LCDs) and their covered ICD-10 codes from the public
CMS Coverage API (https://api.coverage.cms.gov) into data/policies/.

For each policy it writes:
  data/policies/<LCD>.txt        the coverage rules as plain text ("Page 1" = Coverage Indications section,
                                 "Page 2" = Documentation / associated information section)
  data/policies/<LCD>_icd10.json the ICD-10 codes the billing article accepts {code: description}

Which LCD and article to fetch is set in procedures.yaml (cms block). Run: uv run python -m app.ingest.fetch_cms
"""

import html
import json
import re

import httpx
import yaml

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


def latest_version(client: httpx.Client, report: str, id_key: str, doc_id: int) -> int:
    """The API needs a version number; the report lists the current version of every final LCD/article."""
    rows = client.get(f"{API}/reports/{report}", params={"state": "all"}).json()["data"]
    for r in rows:
        if r.get(id_key) == doc_id:
            return r["document_version"]
    raise ValueError(f"{doc_id} not found in {report}")


def main() -> None:
    data = yaml.safe_load(config.PROCEDURES_YAML.read_text())
    out_dir = config.DATA / "policies"
    out_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60) as client:
        token = client.get(f"{API}/metadata/license-agreement").json()["data"][0]["Token"]
        client.headers["Authorization"] = f"Bearer {token}"
        for p in data["procedures"]:
            cms = p["cms"]
            lcd_id = int(cms["policy_id"].lstrip("L"))
            article_id = int(str(cms["article_id"]).lstrip("A"))
            lcd_ver = latest_version(client, "local-coverage-final-lcds", "document_id", lcd_id)
            art_ver = latest_version(client, "local-coverage-articles", "document_id", article_id)

            lcd = client.get(f"{API}/data/lcd", params={"lcdid": lcd_id, "ver": lcd_ver}).json()["data"][0]
            text = (f"REAL CMS POLICY: {lcd['title']} (L{lcd_id}, version {lcd_ver}), downloaded from the CMS "
                    f"Coverage API. Page 1 = Coverage Indications, Page 2 = Associated Information.\n\n"
                    f"=== Page 1 ===\n{html_to_text(lcd.get('indication'))}\n\n"
                    f"=== Page 2 ===\n{html_to_text(lcd.get('associated_info'))}\n")
            (out_dir / f"L{lcd_id}.txt").write_text(text)

            rows = client.get(f"{API}/data/article/icd10-covered",
                              params={"articleid": article_id, "ver": art_ver}).json()["data"]
            codes = {r["icd10_code_id"]: r.get("description", "") for r in rows}
            (out_dir / f"L{lcd_id}_icd10.json").write_text(json.dumps(codes, indent=1, sort_keys=True))
            print(f"CPT {p['cpt']}: L{lcd_id} v{lcd_ver} ({len(text):,} chars), A{article_id} v{art_ver} "
                  f"({len(codes):,} covered ICD-10 codes)")
    print(f"Saved to {out_dir}. Next: review criteria.json, then run build_graph and chunk_index.")


if __name__ == "__main__":
    main()
