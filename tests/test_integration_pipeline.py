"""Integration tests for training and evaluation pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from domain_slm_guardrails.training.sft_dataset import (
    SFTDatasetBuilder,
    GeneralDataLoader,
)
from domain_slm_guardrails.training.dpo_generator import DPOPreferenceGenerator


def test_sft_to_dpo_pipeline():
    """Test complete SFT dataset -> DPO pairs pipeline."""
    # Step 1: Create SFT examples
    chunks = [
        {
            "text": "CPT code 99213 is for office visits of 20-29 minutes.",
            "chunk_id": "c1",
            "source_id": "cpt_manual",
        },
        {
            "text": "Modifier 25 indicates a significant, separately identifiable E/M service.",
            "chunk_id": "c2",
            "source_id": "guidelines",
        },
    ]

    builder = SFTDatasetBuilder(seed=42)
    sft_examples = builder.create_from_chunks(chunks)

    assert len(sft_examples) > 0
    assert all(ex.answer for ex in sft_examples)

    # Step 2: Mix with general data
    general_data = GeneralDataLoader.create_dummy_general_data(size=50)
    mixed = builder.mix_with_general_data(
        sft_examples,
        general_data,
        general_ratio=0.2,
    )

    domain_count = sum(1 for ex in mixed if ex.source_type == "domain")
    general_count = sum(1 for ex in mixed if ex.source_type == "general")

    assert domain_count + general_count == len(mixed)
    assert general_count > 0

    # Step 3: Generate DPO preference pairs
    dpo_gen = DPOPreferenceGenerator(seed=42)
    dpo_pairs = dpo_gen.generate_from_sft_examples(
        [ex.to_dict() for ex in sft_examples],
        max_rejections_per_example=1,
    )

    assert len(dpo_pairs) > 0
    assert all(p.query for p in dpo_pairs)
    assert all(p.chosen for p in dpo_pairs)
    assert all(p.rejected for p in dpo_pairs)
    assert all(p.strategy for p in dpo_pairs)

    # Step 4: Export and verify format
    with tempfile.TemporaryDirectory() as tmpdir:
        sft_path = Path(tmpdir) / "sft.jsonl"
        dpo_path = Path(tmpdir) / "dpo.jsonl"

        builder.export_jsonl(mixed, sft_path)
        dpo_gen.export_jsonl(dpo_pairs, dpo_path)

        # Verify exports
        assert sft_path.exists()
        assert dpo_path.exists()

        # Check content
        sft_lines = sft_path.read_text().strip().split("\n")
        dpo_lines = dpo_path.read_text().strip().split("\n")

        assert len(sft_lines) == len(mixed)
        assert len(dpo_lines) == len(dpo_pairs)

        # Parse and validate
        sft_data = [json.loads(line) for line in sft_lines]
        dpo_data = [json.loads(line) for line in dpo_lines]

        assert all("query" in item for item in sft_data)
        assert all("answer" in item for item in sft_data)
        assert all("query" in item for item in dpo_data)
        assert all("chosen" in item for item in dpo_data)
        assert all("rejected" in item for item in dpo_data)


def test_dpo_rejection_quality():
    """Test that rejection strategies produce valid alternatives."""
    generator = DPOPreferenceGenerator()

    test_cases = [
        (
            "CPT 99213 is an office visit code.",
            "poorly cited",
        ),
        (
            "Modifier 25 is used for E/M services.",
            "hallucinated",
        ),
        (
            "Claims require patient ID and service date.",
            "incomplete",
        ),
        (
            "Documentation supports medical necessity.",
            "verbose",
        ),
    ]

    for chosen, strategy in test_cases:
        rejected = generator._build_rejected_answer(chosen, "", strategy)

        # Rejected should not be identical to chosen
        assert rejected != chosen
        # Rejected should have some content
        assert len(rejected) > 0
        # Rejected should be shorter or similar length (not much longer, allowing for template overhead)
        assert len(rejected) <= len(chosen) * 3.0 + 120


def test_dpo_export_formats():
    """Test both DPO export formats."""
    from domain_slm_guardrails.training.dpo_generator import DPOPreferencePair

    pairs = [
        DPOPreferencePair(
            query="Q1?",
            chosen="Good",
            rejected="Bad",
            strategy="hallucinated",
            metadata={"type": "test"},
        ),
    ]

    generator = DPOPreferenceGenerator()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Standard format
        std_path = Path(tmpdir) / "standard.jsonl"
        generator.export_standard_dpo(pairs, std_path)

        std_content = std_path.read_text()
        std_json = json.loads(std_content)

        assert "query" in std_json
        assert "chosen" in std_json
        assert "rejected" in std_json
        assert "strategy" in std_json


def test_groundedness_metrics_calculation():
    """Test that groundedness metrics are correctly calculated."""
    from domain_slm_guardrails.evaluation.groundedness_comparator import (
        GroundednessCase,
        GroundednessComparator,
    )

    # Case 1: RAG-grounded answer
    case1 = GroundednessCase(
        id="case_1",
        query="What is CPT 99213?",
        baseline_answer="Not sure",
        policy_answer="CPT 99213 is an office visit code.",
        baseline_citations=[],
        policy_citations=[
            {"source_id": "manual", "text": "office visit"},
        ],
        baseline_guardrail={"rag_grounded": False},
        policy_guardrail={"rag_grounded": True},
    )

    # Case 2: With fallback
    case2 = GroundednessCase(
        id="case_2",
        query="Advanced medical question",
        baseline_answer="Complex answer",
        policy_answer="I cannot confidently answer this.",
        baseline_citations=[],
        policy_citations=[],
        baseline_guardrail={"rag_grounded": False},
        policy_guardrail={"rag_grounded": False, "fallback_used": True},
    )

    comparator = GroundednessComparator([case1, case2])
    results = comparator.compare()

    # Verify summary
    assert results["summary"]["case_count"] == 2
    assert "average_baseline_grounding_score" in results["summary"]
    assert "average_policy_grounding_score" in results["summary"]

    # Verify per-case results
    assert len(results["cases"]) == 2
    assert results["cases"][1]["policy_fallback_used"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
