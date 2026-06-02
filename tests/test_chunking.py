from __future__ import annotations

from domain_slm_guardrails.ingestion.chunkers import chunk_page
from domain_slm_guardrails.ingestion.loaders import DocumentPage


def test_chunks_are_stable_and_preserve_metadata():
    page = DocumentPage(
        source_id="doc1",
        source_path="doc1.txt",
        page=12,
        text=" ".join(f"token{i}" for i in range(30)),
    )
    chunks = chunk_page(page, domain="medical_billing", chunk_size_tokens=10, chunk_overlap_tokens=2)

    assert len(chunks) == 4
    assert chunks[0].chunk_id == "medical_billing_doc1_p0012_c001"
    assert chunks[0].source_id == "doc1"
    assert chunks[0].page == 12
    assert chunks[0].token_count == 10


def test_chunk_overlap_works():
    page = DocumentPage(
        source_id="doc",
        source_path="doc.txt",
        page=1,
        text="a b c d e f g h",
    )
    chunks = chunk_page(page, domain="d", chunk_size_tokens=4, chunk_overlap_tokens=1)

    assert chunks[0].text == "a b c d"
    assert chunks[1].text == "d e f g"

