from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "medical_prescription"
LABEL_MD_DIR = RAW_DIR / "openfda_labels"
LABEL_JSON_DIR = RAW_DIR / "source_json" / "openfda_labels"
MANIFEST_PATH = RAW_DIR / "manifest.json"
README_PATH = RAW_DIR / "README.md"
OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"


DEFAULT_DRUGS = [
    "acetaminophen",
    "albuterol",
    "amoxicillin",
    "apixaban",
    "aspirin",
    "atorvastatin",
    "azithromycin",
    "clopidogrel",
    "doxycycline",
    "escitalopram",
    "fluoxetine",
    "furosemide",
    "gabapentin",
    "hydrochlorothiazide",
    "ibuprofen",
    "insulin glargine",
    "levothyroxine",
    "lisinopril",
    "losartan",
    "metformin",
    "metoprolol",
    "morphine",
    "naproxen",
    "omeprazole",
    "ondansetron",
    "prednisone",
    "rosuvastatin",
    "semaglutide",
    "sertraline",
    "warfarin",
]


IMPORTANT_SECTIONS = [
    "boxed_warning",
    "indications_and_usage",
    "dosage_and_administration",
    "dosage_forms_and_strengths",
    "contraindications",
    "warnings_and_cautions",
    "adverse_reactions",
    "drug_interactions",
    "use_in_specific_populations",
    "pregnancy",
    "nursing_mothers",
    "pediatric_use",
    "geriatric_use",
    "overdosage",
    "description",
    "clinical_pharmacology",
    "mechanism_of_action",
    "pharmacodynamics",
    "pharmacokinetics",
    "clinical_studies",
    "patient_medication_information",
    "information_for_patients",
    "medication_guide",
]


SECTION_TITLES = {
    "boxed_warning": "Boxed warning",
    "indications_and_usage": "Indications and usage",
    "dosage_and_administration": "Dosage and administration",
    "dosage_forms_and_strengths": "Dosage forms and strengths",
    "contraindications": "Contraindications",
    "warnings_and_cautions": "Warnings and precautions",
    "adverse_reactions": "Adverse reactions",
    "drug_interactions": "Drug interactions",
    "use_in_specific_populations": "Use in specific populations",
    "pregnancy": "Pregnancy",
    "nursing_mothers": "Nursing mothers",
    "pediatric_use": "Pediatric use",
    "geriatric_use": "Geriatric use",
    "overdosage": "Overdosage",
    "description": "Description",
    "clinical_pharmacology": "Clinical pharmacology",
    "mechanism_of_action": "Mechanism of action",
    "pharmacodynamics": "Pharmacodynamics",
    "pharmacokinetics": "Pharmacokinetics",
    "clinical_studies": "Clinical studies",
    "patient_medication_information": "Patient medication information",
    "information_for_patients": "Information for patients",
    "medication_guide": "Medication guide",
}


@dataclass(frozen=True)
class SourceRecord:
    query_drug: str
    status: str
    markdown_path: str | None
    json_path: str | None
    source_url: str
    spl_id: str | None = None
    spl_set_id: str | None = None
    generic_names: list[str] | None = None
    brand_names: list[str] | None = None
    manufacturer_names: list[str] | None = None
    route: list[str] | None = None
    sections: list[str] | None = None
    reason: str | None = None


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "drug"


def flatten_text(values: Any) -> str:
    if values is None:
        return ""
    if isinstance(values, list):
        return "\n\n".join(str(item).strip() for item in values if str(item).strip())
    return str(values).strip()


def openfda_values(label: dict[str, Any], key: str) -> list[str]:
    values = label.get("openfda", {}).get(key, [])
    if isinstance(values, list):
        return [str(item) for item in values if str(item).strip()]
    if values:
        return [str(values)]
    return []


def build_url(drug: str, limit: int) -> str:
    exact = f'openfda.generic_name:"{drug}"'
    params = {
        "search": exact,
        "limit": str(limit),
    }
    return f"{OPENFDA_LABEL_URL}?{urllib.parse.urlencode(params)}"


def fetch_json(url: str, retries: int = 3) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "domain-slm-guardrails/0.1"})
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            last_error = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def score_label(label: dict[str, Any], query_drug: str) -> int:
    score = 0
    for section in IMPORTANT_SECTIONS:
        text = flatten_text(label.get(section))
        if text:
            score += 2
            score += min(len(text) // 2000, 5)
    product_types = [item.lower() for item in openfda_values(label, "product_type")]
    if any("human prescription" in item for item in product_types):
        score += 8
    if flatten_text(label.get("boxed_warning")):
        score += 4
    if openfda_values(label, "generic_name"):
        score += 2
    generic_names = [item.lower() for item in openfda_values(label, "generic_name")]
    normalized_query = query_drug.lower().strip()
    if normalized_query in generic_names:
        score += 25
    elif any(item.startswith(f"{normalized_query} ") for item in generic_names):
        score += 20
    elif any(normalized_query in item for item in generic_names):
        score += 8
    if " and " not in normalized_query and any(" and " in item for item in generic_names):
        score -= 35
    return score


def best_label(payload: dict[str, Any], query_drug: str) -> dict[str, Any] | None:
    results = payload.get("results", [])
    if not isinstance(results, list) or not results:
        return None
    return max((item for item in results if isinstance(item, dict)), key=lambda item: score_label(item, query_drug), default=None)


def render_markdown(drug: str, label: dict[str, Any], source_url: str, collected_at: str) -> str:
    generic_names = ", ".join(openfda_values(label, "generic_name")) or drug
    brand_names = ", ".join(openfda_values(label, "brand_name")) or "Not specified"
    manufacturers = ", ".join(openfda_values(label, "manufacturer_name")) or "Not specified"
    routes = ", ".join(openfda_values(label, "route")) or "Not specified"
    spl_id = ", ".join(openfda_values(label, "spl_id")) or "Not specified"
    spl_set_id = ", ".join(openfda_values(label, "spl_set_id")) or "Not specified"

    lines = [
        f"# FDA Drug Label: {generic_names}",
        "",
        "This document was generated from the official openFDA drug label API for offline RAG ingestion.",
        "It is source material for cited medication information, not a standalone prescribing protocol.",
        "",
        "## Source metadata",
        "",
        f"- Query drug: {drug}",
        f"- Generic name: {generic_names}",
        f"- Brand name: {brand_names}",
        f"- Manufacturer: {manufacturers}",
        f"- Route: {routes}",
        f"- SPL ID: {spl_id}",
        f"- SPL Set ID: {spl_set_id}",
        f"- Source API URL: {source_url}",
        f"- Collected at UTC: {collected_at}",
        "",
    ]

    for section in IMPORTANT_SECTIONS:
        text = flatten_text(label.get(section))
        if not text:
            continue
        lines.extend([f"## {SECTION_TITLES[section]}", "", text, ""])

    return "\n".join(lines).strip() + "\n"


def write_dataset(drugs: list[str], limit: int, pause_seconds: float) -> list[SourceRecord]:
    LABEL_MD_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_JSON_DIR.mkdir(parents=True, exist_ok=True)
    collected_at = datetime.now(timezone.utc).isoformat()
    records: list[SourceRecord] = []

    for drug in drugs:
        source_url = build_url(drug, limit)
        slug = slugify(drug)
        print(f"Collecting {drug}...", flush=True)
        try:
            payload = fetch_json(source_url)
            label = best_label(payload, drug)
            if label is None:
                records.append(SourceRecord(drug, "missing", None, None, source_url, reason="no_results"))
                continue

            markdown_path = LABEL_MD_DIR / f"{slug}.md"
            json_path = LABEL_JSON_DIR / f"{slug}.json"
            markdown_path.write_text(render_markdown(drug, label, source_url, collected_at), encoding="utf-8")
            json_path.write_text(json.dumps(label, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

            sections = [section for section in IMPORTANT_SECTIONS if flatten_text(label.get(section))]
            records.append(
                SourceRecord(
                    query_drug=drug,
                    status="downloaded",
                    markdown_path=str(markdown_path.relative_to(PROJECT_ROOT)),
                    json_path=str(json_path.relative_to(PROJECT_ROOT)),
                    source_url=source_url,
                    spl_id=", ".join(openfda_values(label, "spl_id")) or None,
                    spl_set_id=", ".join(openfda_values(label, "spl_set_id")) or None,
                    generic_names=openfda_values(label, "generic_name"),
                    brand_names=openfda_values(label, "brand_name"),
                    manufacturer_names=openfda_values(label, "manufacturer_name"),
                    route=openfda_values(label, "route"),
                    sections=sections,
                )
            )
        except urllib.error.HTTPError as exc:
            records.append(SourceRecord(drug, "missing", None, None, source_url, reason=f"http_{exc.code}"))
        except Exception as exc:
            records.append(SourceRecord(drug, "error", None, None, source_url, reason=str(exc)))
        time.sleep(pause_seconds)

    manifest = {
        "domain": "medical_prescription",
        "source": "openFDA drug label API",
        "source_endpoint": OPENFDA_LABEL_URL,
        "collected_at_utc": collected_at,
        "collection_policy": "One highest-section-coverage human prescription label record per configured generic drug query.",
        "drug_count": len(drugs),
        "downloaded_count": sum(1 for record in records if record.status == "downloaded"),
        "records": [asdict(record) for record in records],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    README_PATH.write_text(render_readme(manifest), encoding="utf-8")
    return records


def render_readme(manifest: dict[str, Any]) -> str:
    downloaded = manifest["downloaded_count"]
    total = manifest["drug_count"]
    return f"""# Medical Prescription Raw Corpus

This folder contains official-source medication label records collected for the `medical_prescription` domain.

Source: openFDA Drug Label API
Endpoint: {OPENFDA_LABEL_URL}
Collected at UTC: {manifest["collected_at_utc"]}
Downloaded labels: {downloaded} of {total}

The ingestion-ready files are Markdown documents in `openfda_labels/`.
The original selected API records are preserved in `source_json/openfda_labels/`.
The full provenance manifest is `manifest.json`.

Safety note: these files are evidence sources for cited retrieval and are not a prescribing protocol. The application should continue to block autonomous prescribing and require clinician/pharmacist review for medication decisions.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download official medication label sources for the medical_prescription domain.")
    parser.add_argument("--drug", action="append", dest="drugs", help="Generic drug name to download. Can be repeated.")
    parser.add_argument("--drug-file", type=Path, help="Optional newline-delimited generic drug list.")
    parser.add_argument("--limit", type=int, default=10, help="openFDA records to inspect per drug before selecting the richest label.")
    parser.add_argument("--pause-seconds", type=float, default=0.2, help="Pause between API requests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    drugs = list(DEFAULT_DRUGS)
    if args.drug_file:
        file_drugs = [
            line.strip()
            for line in args.drug_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        drugs = file_drugs
    if args.drugs:
        drugs = args.drugs

    unique_drugs = list(dict.fromkeys(drug.strip().lower() for drug in drugs if drug.strip()))
    records = write_dataset(unique_drugs, limit=args.limit, pause_seconds=args.pause_seconds)
    downloaded = sum(1 for record in records if record.status == "downloaded")
    print(f"Downloaded {downloaded}/{len(records)} official medication labels.")
    print(f"Wrote Markdown corpus to {LABEL_MD_DIR}")
    print(f"Wrote source JSON to {LABEL_JSON_DIR}")
    print(f"Wrote manifest to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
