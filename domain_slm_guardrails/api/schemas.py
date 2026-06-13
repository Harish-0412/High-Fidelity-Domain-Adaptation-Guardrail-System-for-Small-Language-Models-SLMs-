from __future__ import annotations

from typing import Any

try:
    from pydantic import BaseModel, Field
except ModuleNotFoundError:
    _MISSING = object()

    class _DefaultFactory:
        def __init__(self, factory: Any):
            self.factory = factory

        def build(self) -> Any:
            return self.factory()

    def Field(default: Any = _MISSING, **kwargs: Any) -> Any:
        default_factory = kwargs.get("default_factory")
        if default is _MISSING and default_factory is not None:
            return _DefaultFactory(default_factory)
        return default

    class BaseModel:
        """Minimal schema fallback for non-API test environments."""

        def __init__(self, **data: Any):
            annotations: dict[str, Any] = {}
            for cls in reversed(self.__class__.mro()):
                annotations.update(getattr(cls, "__annotations__", {}))
            for name in annotations:
                if name in data:
                    value = data.pop(name)
                else:
                    value = self._field_default(name)
                    if value is _MISSING:
                        raise TypeError(f"missing required field: {name}")
                    if isinstance(value, _DefaultFactory):
                        value = value.build()
                setattr(self, name, value)
            for name, value in data.items():
                setattr(self, name, value)

        def dict(self) -> dict[str, Any]:
            annotations: dict[str, Any] = {}
            for cls in reversed(self.__class__.mro()):
                annotations.update(getattr(cls, "__annotations__", {}))
            return {name: getattr(self, name) for name in annotations}

        def model_dump(self) -> dict[str, Any]:
            return self.dict()

        @classmethod
        def _field_default(cls, name: str) -> Any:
            for owner in cls.mro():
                if name in owner.__dict__:
                    return owner.__dict__[name]
            return _MISSING


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


class AgentToolInfo(BaseModel):
    name: str
    description: str


class AgentExecutionStep(BaseModel):
    step_id: str
    tool_name: str
    action: str
    status: str = "planned"
    reason: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] | list[dict[str, Any]] | str | float | int | bool | None = None
    error: str | None = None


class AgentPlanRequest(QueryRequest):
    include_outputs: bool = Field(default=False)


class AgentPlanResponse(BaseModel):
    domain: str
    query: str
    tools_available: list[AgentToolInfo]
    steps: list[AgentExecutionStep]
    final_answer: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    latency_ms: float | None = None


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
