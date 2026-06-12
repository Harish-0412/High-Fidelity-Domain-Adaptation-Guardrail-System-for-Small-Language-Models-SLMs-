import os
import sys

def main():
    from retrieval.vector_store import try_build_qdrant_store
    
    # Connect to the local qdrant we set up
    url = "http://localhost:6333"
    collection_name = "test_medical_prescription"
    
    print("Building QdrantVectorStore...")
    store = try_build_qdrant_store(url=url, collection_name=collection_name, vector_size=384)
    if not store:
        print("Failed to build QdrantVectorStore.")
        sys.exit(1)
        
    print(f"Ensuring collection '{collection_name}' with payload indexes...")
    store.ensure_collection(recreate=True)
    
    print("Fetching collection info to verify indexes...")
    info = store.client.get_collection(collection_name)
    
    print("\n--- Example Index Creation Output ---")
    print(f"Collection Name: {collection_name}")
    print(f"Vector Count: {info.vectors_count}")
    print(f"Payload Indexes:")
    for field, schema in info.payload_schema.items():
        print(f" - Field: {field}, Schema Type: {schema.data_type.value}")
        
    print("\nValidation complete.")

if __name__ == "__main__":
    main()
