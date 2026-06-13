
import json
from pathlib import Path

chunks_path = Path("data/processed/medical_prescription/chunks.jsonl")

with chunks_path.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        chunk = json.loads(line)
        if "amoxicillin" in chunk.get("text", "").lower():
            print(f"Source: {chunk.get('source_id')}, ID: {chunk.get('chunk_id')}")
            print("Preview:", chunk.get("text")[:300].encode("ascii", errors="replace").decode())
            print()
