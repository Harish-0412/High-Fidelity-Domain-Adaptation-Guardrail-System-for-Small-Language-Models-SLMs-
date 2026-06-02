from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    domain: str = Field(default="medical_billing")
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    output_format: str = Field(default="answer_with_citations")


class Citation(BaseModel):
    citation_id: str
    chunk_id: str
    source_id: str
    page: int | None = None
    score: float
    text: str


class GuardrailStatus(BaseModel):
    rag_grounded: bool
    json_valid: bool = True
    fallback_used: bool = False
    reason: str | None = None


class QueryResponse(BaseModel):
    domain: str
    query: str
    answer: str
    citations: list[Citation]
    guardrail_status: GuardrailStatus
    latency_ms: float

