from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    domain: str = Field(default="medical_prescription")
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
    critic_score: float | None = None


class QueryResponse(BaseModel):
    domain: str
    query: str
    answer: str
    citations: list[Citation]
    guardrail_status: GuardrailStatus
    latency_ms: float


class ThresholdUpdateRequest(BaseModel):
    domain: str
    threshold: float = Field(ge=0.0, le=1.0)


class VoiceAgentWidgetConfig(BaseModel):
    variant: str | None = None
    placement: str | None = None
    avatar: dict[str, object] | None = None
    bg_color: str | None = None
    text_color: str | None = None
    btn_color: str | None = None
    btn_text_color: str | None = None
    border_color: str | None = None


class VoiceAgentConfig(BaseModel):
    agent_id: str
    name: str
    first_message: str | None = None
    supports_text_only: bool = True
    widget: VoiceAgentWidgetConfig = Field(default_factory=VoiceAgentWidgetConfig)


# ---------------------------------------------------------------------------
# Structured Output Schemas
# ---------------------------------------------------------------------------

class StructuredCitation(BaseModel):
    citation_id: str = Field(description="The citation marker, e.g., C1")
    text: str = Field(description="The exact text being cited")


class AnswerWithCitations(BaseModel):
    answer: str = Field(description="The final answered text")
    citations: list[StructuredCitation] = Field(description="List of citations used in the answer")


class InteractionSeverity(BaseModel):
    drug_a: str = Field(description="The first drug name")
    drug_b: str = Field(description="The second drug name")
    severity: str = Field(description="Severity of interaction: Mild, Moderate, or Severe")
    description: str = Field(description="Details of the interaction")


class DrugInteractionReport(BaseModel):
    interactions: list[InteractionSeverity] = Field(description="List of potential drug interactions")
    summary_warning: str = Field(description="Overall warning summary for the physician")


class PrescriptionSummary(BaseModel):
    patient_instructions: str = Field(description="Instructions formatted for the patient")
    dosage_schedule: str = Field(description="When and how to take the medication")
    side_effects: list[str] = Field(description="List of common side effects to watch for")
    requires_followup: bool = Field(description="Whether a follow-up appointment is recommended")
