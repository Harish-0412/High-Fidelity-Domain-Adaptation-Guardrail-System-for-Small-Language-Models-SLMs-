from __future__ import annotations

import json
from pathlib import Path

from domain_slm_guardrails.core.domain_registry import DomainConfig
from domain_slm_guardrails.ingestion.chunkers import DocumentChunk, chunk_page
from domain_slm_guardrails.ingestion.cleaners import clean_page
from domain_slm_guardrails.ingestion.loaders import discover_documents, load_document


def ingest_domain(domain: DomainConfig) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for path in discover_documents(domain.corpus_path):
        for page in load_document(path):
            cleaned = clean_page(page)
            chunks.extend(
                chunk_page(
                    cleaned,
                    domain=domain.domain_id,
                    chunk_size_tokens=int(domain.settings["chunk_size_tokens"]),
                    chunk_overlap_tokens=int(domain.settings["chunk_overlap_tokens"]),
                )
            )
    return chunks


def write_chunks_jsonl(chunks: list[DocumentChunk], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def read_chunks_jsonl(path: str | Path) -> list[dict[str, object]]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Chunk file does not exist: {input_path}")
    chunks: list[dict[str, object]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def run_ingestion(domain: DomainConfig) -> list[DocumentChunk]:
    chunks = ingest_domain(domain)
    write_chunks_jsonl(chunks, domain.chunks_path)
    return chunks

