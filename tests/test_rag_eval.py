from __future__ import annotations

from domain_slm_guardrails.evaluation.rag_eval import (
    RAGEvalCase,
    evaluate_cases,
    summarize_results,
)


def test_rag_eval_scores_cases():
    cases = [
        RAGEvalCase(
            id="modifier_25",
            domain="medical_billing",
            query="When should modifier 25 be used?",
            expected_terms=["modifier 25"],
            expected_source_ids=["sample_modifier_25"],
        )
    ]

    results = evaluate_cases(cases, top_k=3)
    summary = summarize_results(results)

    assert summary["total"] == 1
    assert summary["passed"] == 1
    assert results[0].citation_present is True

