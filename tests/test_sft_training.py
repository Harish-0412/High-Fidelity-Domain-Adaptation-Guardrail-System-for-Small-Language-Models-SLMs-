"""Tests for SFT dataset creation and training."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from services.training.sft_dataset import (
    SFTExample,
    SFTDatasetBuilder,
    GeneralDataLoader,
)


def test_sft_example_serialization():
    """Test SFT example to/from dict conversion."""
    example = SFTExample(
        id="test_1",
        query="What is X?",
        answer="X is Y.",
        citations=[{"source_id": "doc1", "text": "snippet"}],
        source_type="domain",
        metadata={"key": "value"},
    )
    
    data = example.to_dict()
    restored = SFTExample.from_dict(data)
    
    assert restored.id == example.id
    assert restored.query == example.query
    assert restored.answer == example.answer


def test_dataset_builder_from_chunks():
    """Test creating SFT examples from domain chunks."""
    chunks = [
        {
            "text": "CPT code 99213 is for office visits.",
            "chunk_id": "c1",
            "source_id": "cpt_manual",
        },
        {
            "text": "ICD-10 code J45.9 represents asthma.",
            "chunk_id": "c2",
            "source_id": "icd_manual",
        },
    ]
    
    builder = SFTDatasetBuilder()
    examples = builder.create_from_chunks(chunks)
    
    assert len(examples) > 0
    assert all(isinstance(ex, SFTExample) for ex in examples)
    assert all(ex.source_type == "domain" for ex in examples)


def test_dataset_builder_train_val_split():
    """Test train/val split functionality."""
    examples = [SFTExample(
        id=f"ex_{i}",
        query=f"Query {i}",
        answer=f"Answer {i}",
    ) for i in range(100)]
    
    builder = SFTDatasetBuilder()
    train, val = builder.split_train_val(examples, train_ratio=0.8)
    
    assert len(train) == 80
    assert len(val) == 20
    assert len(set(ex.id for ex in train) & set(ex.id for ex in val)) == 0


def test_dataset_builder_export_import():
    """Test JSONL export and import."""
    examples = [
        SFTExample(
            id="ex_1",
            query="Question 1",
            answer="Answer 1",
        ),
        SFTExample(
            id="ex_2",
            query="Question 2",
            answer="Answer 2",
        ),
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "dataset.jsonl"
        builder = SFTDatasetBuilder()
        
        builder.export_jsonl(examples, path)
        loaded = builder.import_jsonl(path)
        
        assert len(loaded) == 2
        assert loaded[0].id == "ex_1"
        assert loaded[1].query == "Question 2"


def test_dataset_mixing():
    """Test mixing domain and general data."""
    domain_examples = [SFTExample(
        id=f"domain_{i}",
        query=f"Domain query {i}",
        answer=f"Domain answer {i}",
        source_type="domain",
    ) for i in range(80)]
    
    general_examples = [SFTExample(
        id=f"general_{i}",
        query=f"General query {i}",
        answer=f"General answer {i}",
        source_type="general",
    ) for i in range(50)]
    
    builder = SFTDatasetBuilder()
    mixed = builder.mix_with_general_data(
        domain_examples,
        general_examples,
        general_ratio=0.2,
    )
    
    # With 20% general ratio and 80 domain examples:
    # Total should be ~100, with ~20 general
    general_count = sum(1 for ex in mixed if ex.source_type == "general")
    assert 15 <= general_count <= 25


def test_general_data_loader_dummy():
    """Test creating dummy general data."""
    data = GeneralDataLoader.create_dummy_general_data(size=50)
    
    assert len(data) == 50
    assert all(isinstance(ex, SFTExample) for ex in data)
    assert all(ex.source_type == "general" for ex in data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
