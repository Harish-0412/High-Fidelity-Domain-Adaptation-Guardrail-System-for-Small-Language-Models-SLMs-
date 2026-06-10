"""Tests for DPO training and preference generation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from domain_slm_guardrails.training.dpo_generator import (
    DPOPreferencePair,
    DPOPreferenceGenerator,
)


def test_dpo_preference_pair_serialization():
    """Test DPO preference pair to/from dict."""
    pair = DPOPreferencePair(
        query="Question?",
        chosen="Good answer.",
        rejected="Bad answer.",
        strategy="hallucinated",
        metadata={"source": "test"},
    )
    
    data = pair.to_dict()
    assert data["query"] == "Question?"
    assert data["strategy"] == "hallucinated"


def test_dpo_preference_generator_from_sft():
    """Test generating preference pairs from SFT examples."""
    sft_examples = [
        {
            "id": "ex_1",
            "query": "What is CPT 99213?",
            "chosen": "CPT 99213 is an established patient office visit code.",
            "citations": [{"source_id": "cpt_manual"}],
        },
        {
            "id": "ex_2",
            "query": "What does modifier 25 mean?",
            "chosen": "Modifier 25 indicates a significant, separately identifiable E/M service.",
            "citations": [{"source_id": "guidelines"}],
        },
    ]
    
    generator = DPOPreferenceGenerator()
    pairs = generator.generate_from_sft_examples(sft_examples)
    
    assert len(pairs) > 0
    assert all(isinstance(p, DPOPreferencePair) for p in pairs)
    assert all(p.query for p in pairs)
    assert all(p.chosen for p in pairs)
    assert all(p.rejected for p in pairs)


def test_dpo_rejection_strategies():
    """Test different rejection strategies."""
    generator = DPOPreferenceGenerator()
    
    chosen = "The answer is that CPT code 99213 is for established patient office visits."
    
    weakly_cited = generator._make_weakly_cited(chosen)
    assert "According to" in weakly_cited
    assert len(weakly_cited) < len(chosen)
    
    hallucinated = generator._make_hallucinated(chosen)
    assert "believed" in hallucinated or "may" in hallucinated
    
    incomplete = generator._make_incomplete(chosen)
    assert len(incomplete) < len(chosen)
    
    verbose = generator._make_overly_verbose(chosen)
    assert len(verbose) > len(chosen)


def test_dpo_export_jsonl():
    """Test exporting preference pairs to JSONL."""
    pairs = [
        DPOPreferencePair(
            query="Q1?",
            chosen="Good answer.",
            rejected="Bad answer.",
            strategy="hallucinated",
        ),
        DPOPreferencePair(
            query="Q2?",
            chosen="Better answer.",
            rejected="Worse answer.",
            strategy="incomplete",
        ),
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "dpo_pairs.jsonl"
        generator = DPOPreferenceGenerator()
        
        result_path = generator.export_jsonl(pairs, path)
        
        assert result_path.exists()
        lines = result_path.read_text().strip().split("\n")
        assert len(lines) == 2


def test_dpo_export_standard_format():
    """Test exporting in standard DPO format."""
    pairs = [
        DPOPreferencePair(
            query="Question?",
            chosen="Chosen response.",
            rejected="Rejected response.",
            strategy="hallucinated",
            metadata={"type": "test"},
        ),
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "standard_dpo.jsonl"
        generator = DPOPreferenceGenerator()
        
        result_path = generator.export_standard_dpo(pairs, path)
        
        assert result_path.exists()
        content = result_path.read_text()
        assert "query" in content
        assert "chosen" in content
        assert "rejected" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
