from __future__ import annotations

from api.rag import answer_query


def test_answer_query_returns_citations():
    response = answer_query(
        domain="medical_prescription",
        query="What is the recommended dosage of albuterol sulfate for acute bronchospasm?",
        top_k=3,
    )

    assert response.guardrail_status.rag_grounded is True
    assert response.citations
    assert response.citations[0].citation_id == "C1"
    assert "[C1]" in response.answer


def test_answer_query_falls_back_for_no_evidence():
    response = answer_query(
        domain="medical_prescription",
        query="zzzznotamedicalprescriptionterm qqqqunknown",
        top_k=3,
    )

    assert response.guardrail_status.fallback_used is True
    assert response.citations == []

