from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from domain_slm_guardrails.api.schemas import Citation


@dataclass
class PromptTemplates:
    """Prompt templates for RAG system with guardrails.
    """

    SYSTEM_PROMPT = """You are a helpful, honest, and accurate assistant for answering domain-specific questions.

Core Guidelines:
1. **Only use information from the provided CONTEXT to answer the question. Do NOT use any external knowledge.
2. **If you don't have enough information to answer, clearly state: "I could not find enough information in the provided context to answer this question."
3. **Citation Format**: When you use information from a context chunk, cite it using [CitationID] format at the end of relevant sentences (e.g., "The guideline states X [C1]").
4. **Answer Structure**: 
   - Start with a direct answer
   - Include citations for every claim you make
   - Keep your answer clear, concise, and directly relevant
5. **No Hallucination**: Do NOT invent, make up, or infer any information not explicitly present in the context.
6. **Domain Specific**: Use terminology and guidelines from the context, no general knowledge.
"""

    USER_PROMPT_TEMPLATE = """--- CONTEXT START ---
{context}
--- CONTEXT END ---

--- QUESTION START ---
{question}
--- QUESTION END ---

Please answer the question using only information from the CONTEXT. Follow all guidelines, especially citing every claim with the relevant [CitationID].
"""

    FALLBACK_PROMPT = """You are a helpful assistant that will provide a fallback response when the primary generation fails or is not available.

Guidelines:
1. Be polite and helpful
2. State clearly that we're having trouble generating an answer right now
3. Invite the user to try again or rephrase their question
"""

    @classmethod
    def build_context(cls, citations: Sequence[Citation]) -> str:
        """Build context string from list of citations."""
        context_parts = []
        for citation in citations:
            metadata = []
            if citation.source_id:
                metadata.append(f"Source: {citation.source_id}")
            if citation.page:
                metadata.append(f"Page: {citation.page}")
            if citation.score:
                metadata.append(f"Relevance Score: {citation.score:.4f}")
            metadata_str = f" ({', '.join(metadata)})" if metadata else ""
            context_parts.append(
                f"[Citation {citation.citation_id}]{metadata_str}:\n{citation.text}"
            )
        return "\n\n".join(context_parts)

    @classmethod
    def build_user_prompt(cls, question: str, citations: Sequence[Citation]) -> str:
        """Build complete user prompt from question and citations."""
        context = cls.build_context(citations)
        return cls.USER_PROMPT_TEMPLATE.format(context=context, question=question)

    @classmethod
    def build_fallback_prompt(cls) -> str:
        """Build fallback prompt for when generation fails."""
        return cls.FALLBACK_PROMPT
