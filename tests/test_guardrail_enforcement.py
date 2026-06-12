from __future__ import annotations

import json
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.rag import enforcer, answer_query
from services.critic.enforcer import LiveGuardrailEnforcer

client = TestClient(app)


def test_enforcer_fallback_jaccard_scoring():
    # Instantiate enforcer without model checkpoint to trigger text-based Jaccard path
    test_enforcer = LiveGuardrailEnforcer()
    
    # Grounded answer: highly overlaps with context
    query = "dosage of albuterol"
    context = "Albuterol sulfate dosage: two inhalations every 4 hours."
    answer = "two inhalations every 4 hours"
    
    res = test_enforcer.score_and_enforce(
        query=query,
        retrieved_context=context,
        generated_answer=answer,
        domain="medical_prescription",
        threshold=0.4,
    )
    
    assert res["critic_score"] == 0.0
    assert res["fallback_used"] is False
    assert res["reason"] is None

    # Hallucinated answer: no overlap with context
    hallucinated_answer = "Take three tablets of amoxicillin daily."
    res_hallucinated = test_enforcer.score_and_enforce(
        query=query,
        retrieved_context=context,
        generated_answer=hallucinated_answer,
        domain="medical_prescription",
        threshold=0.4,
    )
    
    assert res_hallucinated["critic_score"] == 1.0
    assert res_hallucinated["fallback_used"] is True
    assert res_hallucinated["reason"] == "critic_threshold_crossed"


def test_enforcer_audit_logging():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_enforcer = LiveGuardrailEnforcer()
        log_file = Path(tmpdir) / "audit.log"
        test_enforcer.audit_log_path = log_file

        # Trigger score check which writes logs
        test_enforcer.score_and_enforce(
            query="test query",
            retrieved_context="source context text",
            generated_answer="grounded text",
            domain="medical_prescription",
            threshold=0.5,
        )

        assert log_file.exists()
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 1
        
        log_data = json.loads(lines[0])
        assert log_data["domain"] == "medical_prescription"
        assert log_data["query"] == "test query"
        assert log_data["action_taken"] in ("continue", "fallback")
        assert "timestamp" in log_data


def test_enforcer_telemetry_metrics():
    test_enforcer = LiveGuardrailEnforcer()
    test_enforcer.reset_metrics()

    # 1. First query is grounded (score = 0.0)
    test_enforcer.score_and_enforce(
        query="q1",
        retrieved_context="word1 word2",
        generated_answer="word1 word2",
        domain="medical_prescription",
        threshold=0.4,
    )

    # 2. Second query is hallucinated (score = 1.0)
    test_enforcer.score_and_enforce(
        query="q2",
        retrieved_context="word1 word2",
        generated_answer="different word",
        domain="medical_prescription",
        threshold=0.4,
    )

    metrics = test_enforcer.get_metrics()
    assert metrics["total_queries"] == 2
    assert metrics["total_fallbacks"] == 1
    assert metrics["fallback_rate"] == 0.5
    assert metrics["average_critic_score"] == 0.5

    # Test reset
    test_enforcer.reset_metrics()
    reset_metrics = test_enforcer.get_metrics()
    assert reset_metrics["total_queries"] == 0
    assert reset_metrics["total_fallbacks"] == 0
    assert reset_metrics["average_critic_score"] == 0.0


def test_api_thresholds_endpoint():
    # Verify current thresholds
    resp = client.get("/guardrail/thresholds")
    assert resp.status_code == 200
    data = resp.json()
    assert "medical_prescription" in data
    # Threshold from domain.yaml should be 0.4
    assert data["medical_prescription"] == 0.4

    # Update threshold override dynamically
    update_resp = client.post(
        "/guardrail/thresholds",
        json={"domain": "medical_prescription", "threshold": 0.25},
    )
    assert update_resp.status_code == 200
    assert update_resp.json()["threshold"] == 0.25

    # Check updated threshold
    resp2 = client.get("/guardrail/thresholds")
    assert resp2.json()["medical_prescription"] == 0.25

    # Reset enforcer override for subsequent tests
    enforcer.runtime_threshold_overrides.clear()


def test_api_metrics_and_logs_endpoints():
    # Reset metrics first
    client.post("/guardrail/metrics/reset")

    # Force query to update metrics
    query_resp = client.post(
        "/query",
        json={
            "domain": "medical_prescription",
            "query": "What is the recommended dosage of albuterol sulfate for acute bronchospasm?",
            "top_k": 3,
        },
    )
    assert query_resp.status_code == 200

    # Retrieve metrics
    metrics_resp = client.get("/guardrail/metrics")
    assert metrics_resp.status_code == 200
    metrics = metrics_resp.json()
    assert metrics["total_queries"] >= 1

    # Retrieve logs
    logs_resp = client.get("/guardrail/logs?limit=5")
    assert logs_resp.status_code == 200
    logs = logs_resp.json()
    assert len(logs) >= 1
    assert logs[-1]["domain"] == "medical_prescription"


@pytest.mark.skipif(enforcer.critic_model is None, reason="PyTorch Critic Model is not loaded")
def test_pytorch_tensor_scoring():
    # If the Critic model is loaded, we can verify direct PyTorch tensor scoring
    import torch
    
    hidden_size = enforcer.critic_metadata["hidden_size"]
    test_tensor = torch.randn((1, 5, hidden_size))
    score = enforcer.score_sequence_tensor(test_tensor)
    
    assert 0.0 <= score <= 1.0
