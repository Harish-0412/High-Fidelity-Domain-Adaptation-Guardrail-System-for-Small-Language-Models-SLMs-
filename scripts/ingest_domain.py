from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.core.domain_registry import get_domain_config
from ingestion.pipeline import run_ingestion


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest raw domain documents into JSONL chunks.")
    parser.add_argument("--domain", required=True, help="Domain id, for example medical_prescription")
    args = parser.parse_args()

    domain = get_domain_config(args.domain)
    chunks = run_ingestion(domain)
    print(f"Ingested {len(chunks)} chunks for domain '{domain.domain_id}'.")
    print(f"Wrote chunks to {domain.chunks_path}")


if __name__ == "__main__":
    main()
