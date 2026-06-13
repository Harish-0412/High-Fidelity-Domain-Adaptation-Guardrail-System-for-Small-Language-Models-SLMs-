from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class HallucinationCheckResult:
    has_hallucinations: bool
    confidence: float
    flagged_segments: List[str]
    explanation: str


class HallucinationDetector:
    """Hallucination detector by comparing LLM outputs to retrieved context."""

    def __init__(self, strictness: float = 0.7):
        self.strictness = strictness

    def check(
        self,
        generated_response: str,
        retrieved_chunks: List[dict],
    ) -> HallucinationCheckResult:
        if not retrieved_chunks:
            return HallucinationCheckResult(
                has_hallucinations=True,
                confidence=0.9,
                flagged_segments=["No context available to verify response"],
                explanation="Response generated without supporting context",
            )

        context_text = " ".join([chunk.get("text", "") for chunk in retrieved_chunks])
        context_tokens = set(self._tokenize(context_text.lower()))
        response_tokens = self._tokenize(generated_response.lower())
        response_sentences = self._split_sentences(generated_response)

        flagged_segments = []
        for sent in response_sentences:
            sent_tokens = set(self._tokenize(sent.lower()))
            if not sent_tokens:
                continue
            overlap = len(sent_tokens & context_tokens) / len(sent_tokens)
            if overlap < self.strictness:
                # Check if it's a simple factual statement not in context
                if self._is_factual_statement(sent):
                    flagged_segments.append(sent)

        has_hallucinations = len(flagged_segments) > 0
        confidence = min(1.0, (len(flagged_segments) / max(len(response_sentences), 1)) * 2)

        explanation = (
            f"Response had {len(flagged_segments)} potentially hallucinated statements"
            if has_hallucinations
            else "Response seems well-supported by context"
        )

        return HallucinationCheckResult(
            has_hallucinations=has_hallucinations,
            confidence=confidence,
            flagged_segments=flagged_segments,
            explanation=explanation,
        )

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text)

    def _split_sentences(self, text: str) -> List[str]:
        return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

    def _is_factual_statement(self, text: str) -> bool:
        # Heuristic: exclude questions, commands
        text = text.strip()
        if text.endswith("?") or text.lower().startswith(("tell", "explain", "how", "what", "why")):
            return False
        return True
