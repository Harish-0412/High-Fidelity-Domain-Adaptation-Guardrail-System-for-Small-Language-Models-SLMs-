from __future__ import annotations

from domain_slm_guardrails.evaluation.rag_eval import (
    RAGEvalCase,
    evaluate_cases,
    summarize_results,
)


def test_rag_eval_scores_cases():
    cases = [
        RAGEvalCase(
            id="albuterol_dosage",
            domain="medical_prescription",
            query="What is the recommended dosage of albuterol sulfate for acute bronchospasm?",
            expected_terms=["two inhalations", "4 to 6 hours"],
            expected_source_ids=["albuterol"],
        )
    ]

    results = evaluate_cases(cases, top_k=3)
    summary = summarize_results(results)

    assert summary["total"] == 1
    assert summary["passed"] == 1
    assert results[0].citation_present is True

