#!/usr/bin/env python3
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieval.vector_store import try_build_qdrant_store
from services.core.domain_registry import get_domain_config

def main():
    domain = get_domain_config("medical_prescription")
    store = try_build_qdrant_store("http://localhost:6333", domain.index_name, 384)
    if not store:
        sys.exit(1)
        
    info = store.client.get_collection(domain.index_name)
    
    print("\n# VECTOR COUNT")
    print(getattr(info, 'points_count', getattr(info, 'vectors_count', 'unknown')))
    
    print("\n# SAMPLE PAYLOADS")
    records, _ = store.client.scroll(
        collection_name=domain.index_name,
        limit=5,
        with_payload=True,
        with_vectors=False
    )
    for i, r in enumerate(records):
        # Truncate text for readability
        payload = dict(r.payload) if r.payload else {}
        if 'text' in payload and isinstance(payload['text'], str):
            payload['text'] = payload['text'][:100] + "... [TRUNCATED]"
        print(f"--- Payload {i+1} ---")
        print(json.dumps(payload, indent=2))
        
    print("\n# PAYLOAD INDEXES")
    if hasattr(info, 'payload_schema') and info.payload_schema:
        for field, schema in info.payload_schema.items():
            print(f"- {field}: {schema.data_type.value}")
    else:
        print("None found.")
        
    print("\n# COLLECTION CONFIGURATION")
    print(f"Name: {domain.index_name}")
    print(f"Status: {info.status.value}")
    # Fix for Pydantic V1/V2 compatibility
    vectors_config = info.config.params.vectors if info.config else None
    if vectors_config:
        print(f"Dimension: {getattr(vectors_config, 'size', 'unknown')}")
        print(f"Distance: {getattr(vectors_config, 'distance', 'unknown')}")
    else:
        print("Vectors Config: Unknown")
        

if __name__ == "__main__":
    main()
