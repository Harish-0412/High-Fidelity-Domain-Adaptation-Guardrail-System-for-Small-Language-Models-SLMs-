from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Sequence, Tuple

from domain_slm_guardrails.api.schemas import Citation
from domain_slm_guardrails.llm import LLMConfig, OllamaClient, PromptTemplates

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationConfig:
    min_confidence_for_generation: float = 0.05
    citation_regex: str = r"\[C(\d+)\]"
    max_answer_length: int = 2000
    enable_citation_validation: bool = True
    enable_answer_sanitization: bool = True


@dataclass
class GenerationResult:
    answer: str
    citations_used: list[str]
    was_fallback: bool = False
    error: str | None = None


class RAGGenerator:
    def __init__(
        self,
        llm_client: OllamaClient | None = None,
        generation_config: GenerationConfig | None = None,
        llm_config: LLMConfig | None = None,
    ):
        self.llm_client = llm_client or OllamaClient(llm_config)
        self.generation_config = generation_config or GenerationConfig()
        logger.info(
            f"RAGGenerator initialized with model: {self.llm_client.config.model}"
        )

    def generate_answer(
        self,
        question: str,
        citations: Sequence[Citation],
    ) -> GenerationResult:
        """Generate answer with RAG using LLM.

        Args:
            question: User's question
            citations: Retrieved citations to use as context

        Returns:
            GenerationResult with answer, citations used, and metadata
        """
        if not citations:
            logger.warning("No citations provided for generation")
            return GenerationResult(
                answer="I could not find any relevant information to answer your question.",
                citations_used=[],
                was_fallback=True,
            )

        try:
            logger.info(f"Generating answer for question: {question[:100]}...")
            user_prompt = PromptTemplates.build_user_prompt(question, citations)
            answer = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=PromptTemplates.SYSTEM_PROMPT,
            )

            # Sanitize and validate the answer
            if self.generation_config.enable_answer_sanitization:
                answer = self._sanitize_answer(answer)

            citations_used = []
            if self.generation_config.enable_citation_validation:
                _, citations_used = self.validate_citations(answer, citations)
                logger.info(f"Valid citations found in answer: {citations_used}")

            return GenerationResult(
                answer=answer,
                citations_used=citations_used,
                was_fallback=False,
            )
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}", exc_info=True)
            fallback_answer = self._get_extractive_fallback(question, citations)
            return GenerationResult(
                answer=fallback_answer,
                citations_used=[c.citation_id for c in citations[:3]],
                was_fallback=True,
                error=str(e),
            )

    def _sanitize_answer(self, answer: str) -> str:
        """Clean and sanitize generated answer."""
        # Remove extra whitespace
        answer = re.sub(r"\s+", " ", answer).strip()
        # Truncate if too long
        if len(answer) > self.generation_config.max_answer_length:
            answer = answer[: self.generation_config.max_answer_length]
            # Try to truncate at last sentence
            last_period = answer.rfind(".")
            if last_period > self.generation_config.max_answer_length // 2:
                answer = answer[: last_period + 1]
        return answer

    def _get_extractive_fallback(
        self, question: str, citations: Sequence[Citation]
    ) -> str:
        """Get a simple extractive fallback answer."""
        if not citations:
            return "I could not find any relevant information to answer your question."

        # Take top 3 citations as fallback
        fallback_parts = [
            "Based on the available information:",
        ]
        for i, citation in enumerate(citations[:3], 1):
            fallback_parts.append(f"{i}. {citation.text[:300]}... [{citation.citation_id}]")

        return "\n\n".join(fallback_parts)

    def extract_citation_ids(self, answer: str) -> list[str]:
        """Extract citation IDs from answer text."""
        matches = re.findall(self.generation_config.citation_regex, answer)
        return [f"C{m}" for m in matches]

    def validate_citations(
        self, answer: str, valid_citations: Sequence[Citation]
    ) -> Tuple[str, list[str]]:
        """Validate and clean citations in the answer.

        Args:
            answer: Generated answer text
            valid_citations: List of valid citations that should be referenced

        Returns:
            Tuple of (cleaned_answer, list_of_valid_citation_ids)
        """
        valid_ids = {c.citation_id for c in valid_citations}
        extracted_ids = self.extract_citation_ids(answer)
        valid_extracted = []
        cleaned_answer = answer

        # Replace invalid citations with nothing
        for cid in extracted_ids:
            if cid in valid_ids:
                valid_extracted.append(cid)
            else:
                cleaned_answer = cleaned_answer.replace(f"[{cid}]", "")

        return cleaned_answer, valid_extracted
