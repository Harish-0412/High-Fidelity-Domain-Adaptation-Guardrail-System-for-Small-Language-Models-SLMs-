#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.rag import answer_query

def main():
    print("Testing Ollama Generation Integration...")
    
    # Query 1: Amoxicillin dosage
    print("\n--- Query 1: Amoxicillin Dosage ---")
    resp = answer_query(
        domain="medical_prescription", 
        query="What is the dosage of amoxicillin?", 
        top_k=5, 
        output_format="answer_with_citations"
    )
    print("Answer:")
    print(resp.answer)
    print("Status:", resp.guardrail_status)
    print("Latency:", resp.latency_ms, "ms")

    # Query 2: Aspirin (OOD Guardrail test)
    print("\n--- Query 2: Aspirin Side Effects (OOD Guardrail Test) ---")
    resp = answer_query(
        domain="medical_prescription", 
        query="What are the side effects of aspirin?", 
        top_k=5, 
        output_format="answer_with_citations"
    )
    print("Answer:")
    print(resp.answer)
    print("Status:", resp.guardrail_status)
    print("Latency:", resp.latency_ms, "ms")

if __name__ == "__main__":
    main()
