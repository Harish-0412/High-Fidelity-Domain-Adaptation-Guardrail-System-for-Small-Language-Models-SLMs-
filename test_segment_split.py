#!/usr/bin/env python3
from domain_slm_guardrails.api.rag import _split_segments
import json

with open("data/processed/medications/chunks.jsonl", "r") as f:
    for line in f:
        chunk = json.loads(line)
        if chunk["chunk_id"] == "medications_common_medications_and_conditions_p0001_c001":
            text = chunk["text"]
            print("Testing segment split on chunk 001:")
            print()
            segments = list(_split_segments(text))
            print(f"Found {len(segments)} segments:")
            for idx, seg in enumerate(segments, 1):
                print(f"{idx}) {seg[:100]}")
            break
