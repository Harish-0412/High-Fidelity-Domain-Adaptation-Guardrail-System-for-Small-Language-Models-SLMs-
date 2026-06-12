from __future__ import annotations

from retrieval.embeddings import cosine_similarity

def compute_mmr(
    query_vector: list[float],
    candidate_vectors: list[list[float]],
    candidate_indices: list[int],
    top_k: int = 5,
    lambda_mult: float = 0.5,
) -> list[int]:
    """
    Calculate Maximal Marginal Relevance (MMR) to maximize both relevance and diversity.
    
    Args:
        query_vector: The dense embedding of the search query.
        candidate_vectors: A list of dense embeddings for the candidate chunks.
        candidate_indices: The original ranking or unique IDs of the candidates.
        top_k: The number of results to return.
        lambda_mult: Controls the trade-off between relevance and diversity.
                     1.0 corresponds to standard relevance-only ranking.
                     0.0 corresponds to maximum diversity.
                     
    Returns:
        List of selected candidate indices in MMR-ranked order.
    """
    if not candidate_vectors or top_k <= 0:
        return []
        
    # Precompute relevance scores (cosine similarity to query)
    relevance_scores = [cosine_similarity(query_vector, vec) for vec in candidate_vectors]
    
    selected_idx = []
    unselected_idx = list(range(len(candidate_vectors)))
    
    # First, select the candidate with the highest relevance
    first_sel = max(unselected_idx, key=lambda i: relevance_scores[i])
    selected_idx.append(first_sel)
    unselected_idx.remove(first_sel)
    
    # Iteratively select the remaining candidates
    while len(selected_idx) < top_k and unselected_idx:
        best_score = -float("inf")
        best_idx = -1
        
        for idx in unselected_idx:
            # Find the maximum similarity between this candidate and all already selected candidates
            max_sim_to_selected = max(
                cosine_similarity(candidate_vectors[idx], candidate_vectors[sel_idx])
                for sel_idx in selected_idx
            )
            
            # MMR Equation: λ * relevance - (1 - λ) * redundancy
            mmr_score = lambda_mult * relevance_scores[idx] - (1.0 - lambda_mult) * max_sim_to_selected
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
                
        selected_idx.append(best_idx)
        unselected_idx.remove(best_idx)
        
    return [candidate_indices[i] for i in selected_idx]
