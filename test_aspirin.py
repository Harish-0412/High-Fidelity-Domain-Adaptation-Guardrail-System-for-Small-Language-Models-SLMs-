
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from domain_slm_guardrails.api.rag import answer_query, RAGConfig
from domain_slm_guardrails.retrieval.hybrid import load_hybrid_retriever

# Clear the lru cache of load_hybrid_retriever!
load_hybrid_retriever.cache_clear()

print("Calling answer_query with query: \"what is aspirin?\"")
response = answer_query(
    domain="medical_prescription",
    query="what is aspirin?",
    top_k=15,
    config=RAGConfig(
        max_answer_sentences=3,
        use_mmr=False,
        coherence_sort=False
    )
)

# Print safely
print("=== Answer ===")
safe_answer = response.answer[:2000].encode("ascii", errors="replace").decode()
print(safe_answer)
