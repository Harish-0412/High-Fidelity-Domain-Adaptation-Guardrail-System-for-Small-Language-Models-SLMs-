from __future__ import annotations

from dataclasses import dataclass, asdict
import re

from ingestion.loaders import DocumentPage


TOKEN_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    domain: str
    source_id: str
    source_path: str
    page: int
    text: str
    token_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def chunk_page(
    page: DocumentPage,
    domain: str,
    chunk_size_tokens: int = 512,
    chunk_overlap_tokens: int = 80,
) -> list[DocumentChunk]:
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be positive")
    if chunk_overlap_tokens < 0:
        raise ValueError("chunk_overlap_tokens cannot be negative")
    if chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")

    tokens = tokenize(page.text)
    if not tokens:
        return []

    chunks: list[DocumentChunk] = []
    step = chunk_size_tokens - chunk_overlap_tokens
    chunk_index = 0
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size_tokens]
        if not window:
            continue
        chunk_index += 1
        chunk_id = f"{domain}_{page.source_id}_p{page.page:04d}_c{chunk_index:03d}"
        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                domain=domain,
                source_id=page.source_id,
                source_path=page.source_path,
                page=page.page,
                text=" ".join(window),
                token_count=len(window),
            )
        )
        if start + chunk_size_tokens >= len(tokens):
            break
    return chunks

