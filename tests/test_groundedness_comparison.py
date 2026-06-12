"""Tests for groundedness comparison evaluation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from services.evaluation.groundedness_comparator import (
    GroundednessCase,
    GroundednessComparator,
)


def test_groundedness_case_creation():
    """Test creating groundedness comparison cases."""
    case = GroundednessCase(
        id="case_1",
        query="What is CPT 99213?",
        baseline_answer="It is a billing code.",
        policy_answer="CPT 99213 is an established patient office visit code.",
        baseline_citations=[],
        policy_citations=[{"source_id": "cpt_manual", "text": "99213: office visit"}],
        baseline_guardrail={"rag_grounded": False},
        policy_guardrail={"rag_grounded": True, "critic_score": 0.15},
    )
    
    assert case.id == "case_1"
    assert case.policy_guardrail["rag_grounded"] is True


def test_groundedness_comparator_basic():
    """Test basic groundedness comparison."""
    cases = [
        GroundednessCase(
            id="case_1",
            query="Question 1?",
            baseline_answer="Baseline answer.",
            policy_answer="Policy answer with more detail.",
            baseline_citations=[],
            policy_citations=[{"source_id": "doc1"}],
            baseline_guardrail={"rag_grounded": False},
            policy_guardrail={"rag_grounded": True},
        ),
    ]
    
    comparator = GroundednessComparator(cases)
    results = comparator.compare()
    
    assert "summary" in results
    assert "cases" in results
    assert results["summary"]["case_count"] == 1


def test_groundedness_comparator_multiple_cases():
    """Test comparison with multiple cases."""
    cases = [
        GroundednessCase(
            id=f"case_{i}",
            query=f"Question {i}?",
            baseline_answer=f"Baseline answer {i}",
            policy_answer=f"Policy answer {i} with citations",
            baseline_citations=[],
            policy_citations=[{"source_id": f"doc_{i}"}],
            baseline_guardrail={"rag_grounded": False},
            policy_guardrail={"rag_grounded": True, "critic_score": 0.1 * i},
        )
        for i in range(1, 4)
    ]
    
    comparator = GroundednessComparator(cases)
    results = comparator.compare()
    
    assert results["summary"]["case_count"] == 3
    assert len(results["cases"]) == 3


def test_groundedness_export_json():
    """Test exporting results as JSON."""
    cases = [
        GroundednessCase(
            id="case_1",
            query="Q?",
            baseline_answer="Baseline.",
            policy_answer="Policy.",
            baseline_citations=[],
            policy_citations=[],
            baseline_guardrail={"rag_grounded": False},
            policy_guardrail={"rag_grounded": True},
        ),
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "comparison.json"
        comparator = GroundednessComparator(cases)
        result_path = comparator.export_json(path)
        
        assert result_path.exists()
        content = result_path.read_text()
        assert "summary" in content


def test_groundedness_export_csv():
    """Test exporting results as CSV."""
    cases = [
        GroundednessCase(
            id="case_1",
            query="Q?",
            baseline_answer="Baseline.",
            policy_answer="Policy.",
            baseline_citations=[],
            policy_citations=[],
            baseline_guardrail={"rag_grounded": False},
            policy_guardrail={"rag_grounded": True},
        ),
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "comparison.csv"
        comparator = GroundednessComparator(cases)
        result_path = comparator.export_csv(path)
        
        assert result_path.exists()
        content = result_path.read_text()
        assert "case_1" in content


def test_groundedness_export_markdown():
    """Test exporting results as Markdown."""
    cases = [
        GroundednessCase(
            id="case_1",
            query="Q?",
            baseline_answer="Baseline.",
            policy_answer="Policy.",
            baseline_citations=[],
            policy_citations=[],
            baseline_guardrail={"rag_grounded": False},
            policy_guardrail={"rag_grounded": True},
        ),
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "comparison.md"
        comparator = GroundednessComparator(cases)
        result_path = comparator.export_markdown(path)
        
        assert result_path.exists()
        content = result_path.read_text()
        assert "case_1" in content
        assert "|" in content  # Check for table format


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
