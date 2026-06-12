import os
import sys

def main():
    print("Testing imports...")
    try:
        import fitz
        print("PyMuPDF (fitz) is installed.")
    except ImportError:
        print("MISSING: PyMuPDF (fitz)")

    try:
        from sentence_transformers import SentenceTransformer
        print("sentence-transformers is installed.")
    except ImportError:
        print("MISSING: sentence-transformers")

    try:
        from qdrant_client import QdrantClient
        print("qdrant-client is installed.")
    except ImportError:
        print("MISSING: qdrant-client")

    print("\nTesting loaders...")
    from ingestion.loaders import load_document, discover_documents
    print("Loaders imported successfully.")

    print("\nTesting chunkers...")
    from ingestion.chunkers import chunk_page
    print("Chunkers imported successfully.")

    print("\nTesting vector_store...")
    from retrieval.vector_store import QdrantVectorStore
    print("VectorStore imported successfully.")

    print("\nTesting embeddings...")
    from retrieval.embeddings import load_embedding_model
    print("Embeddings imported successfully.")

if __name__ == "__main__":
    main()
