from __future__ import annotations

import logging
from typing import Optional

from domain_slm_guardrails.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("ollama package not installed. Ollama client will not be available.")


class OllamaClient(BaseLLMClient):
    """Ollama LLM client for local model inference"""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        """
        Initialize Ollama client.

        Args:
            model: Model name to use (e.g., "llama3.2", "mistral", "phi3")
            base_url: Ollama server URL
            timeout: Request timeout in seconds
        """
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "ollama package is not installed. "
                "Install it with: pip install ollama"
            )

        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._client = ollama.Client(host=base_url)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate text from the given prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters for Ollama

        Returns:
            Generated text
        """
        try:
            response = self._client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    **kwargs
                }
            )
            return response.get("response", "").strip()
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise

    def generate_with_context(
        self,
        query: str,
        context: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate text with RAG context.

        Args:
            query: User's query
            context: Retrieved context/documents
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters for Ollama

        Returns:
            Generated answer
        """
        prompt = self._build_rag_prompt(query, context)
        return self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

    def _build_rag_prompt(self, query: str, context: str) -> str:
        """Build a RAG prompt with context."""
        return f"""You are a medical prescription assistant. Use the following context to answer the user's question accurately and concisely. If the context doesn't contain enough information to answer the question, state that clearly.

Context:
{context}

Question: {query}

Answer:"""

    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            self._client.list()
            return True
        except Exception as e:
            logger.warning(f"Ollama server not available: {e}")
            return False

    def list_models(self) -> list[str]:
        """List available models on the Ollama server."""
        try:
            models = self._client.list()
            return [model.get("name", "") for model in models.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []
