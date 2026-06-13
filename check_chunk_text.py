#!/usr/bin/env python3
import json

with open("data/processed/medications/chunks.jsonl", "r") as f:
    for line in f:
        chunk = json.loads(line)
        print(repr(chunk["text"][:500]))
        print()
        break
