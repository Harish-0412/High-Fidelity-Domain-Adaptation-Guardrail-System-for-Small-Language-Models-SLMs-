from __future__ import annotations

from domain_slm_guardrails.api.rag import answer_query


def test_answer_query_returns_citations():
    response = answer_query(
        domain="medical_billing",
        query="When should modifier 25 be used?",
        top_k=3,
    )

    assert response.guardrail_status.rag_grounded is True
    assert response.citations
    assert response.citations[0].citation_id == "C1"
    assert "[C1]" in response.answer


def test_answer_query_falls_back_for_no_evidence():
    response = answer_query(
        domain="medical_billing",
        query="zzzznotamedicalbillingterm qqqqunknown",
        top_k=3,
    )

    assert response.guardrail_status.fallback_used is True
    assert response.citations == []

