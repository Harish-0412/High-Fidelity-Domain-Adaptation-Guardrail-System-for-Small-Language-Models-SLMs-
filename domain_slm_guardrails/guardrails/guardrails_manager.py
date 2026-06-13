from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from domain_slm_guardrails.guardrails.hallucination_detector import (
    HallucinationDetector,
    HallucinationCheckResult,
)
from domain_slm_guardrails.guardrails.content_moderator import (
    ContentModerator,
    ContentModerationResult,
)


@dataclass
class GuardrailsResult:
    overall_pass: bool
    hallucination_check: Optional[HallucinationCheckResult] = None
    content_check: Optional[ContentModerationResult] = None
    confidence_score: float = 0.0
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class GuardrailsManager:
    """Manages all guardrails checks for LLM responses."""

    def __init__(
        self,
        hallucination_detector: Optional[HallucinationDetector] = None,
        content_moderator: Optional[ContentModerator] = None,
    ):
        self.hallucination_detector = hallucination_detector or HallucinationDetector()
        self.content_moderator = content_moderator or ContentModerator()

    def apply_guardrails(
        self,
        query: str,
        generated_response: str,
        retrieved_chunks: List[dict],
        skip_hallucination: bool = False,
        skip_content_check: bool = False,
    ) -> GuardrailsResult:
        warnings = []
        suggestions = []
        hallucination_result = None
        content_result = None
        overall_pass = True

        # Content moderation first (check user input)
        if not skip_content_check:
            content_result = self.content_moderator.check(query + " " + generated_response)
            if not content_result.is_safe:
                overall_pass = False
                warnings.append(content_result.explanation)

        # Hallucination check
        if not skip_hallucination:
            hallucination_result = self.hallucination_detector.check(
                generated_response, retrieved_chunks
            )
            if hallucination_result.has_hallucinations:
                overall_pass = False
                warnings.append(hallucination_result.explanation)

        # Calculate confidence
        confidences = []
        if content_result:
            confidences.append(content_result.confidence)
        if hallucination_result:
            confidences.append(1.0 - hallucination_result.confidence)  # reverse for hallucination
        confidence_score = sum(confidences) / max(len(confidences), 1) if confidences else 0.5

        # Add suggestions
        if not retrieved_chunks:
            suggestions.append("No context found. Consider rephrasing your question.")
        if hallucination_result and hallucination_result.has_hallucinations:
            suggestions.append("Some claims may not be supported by available documents.")

        return GuardrailsResult(
            overall_pass=overall_pass,
            hallucination_check=hallucination_result,
            content_check=content_result,
            confidence_score=confidence_score,
            warnings=warnings,
            suggestions=suggestions,
        )
