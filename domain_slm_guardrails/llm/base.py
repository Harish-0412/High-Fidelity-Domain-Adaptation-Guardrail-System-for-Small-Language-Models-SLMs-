from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLMClient(ABC):
    """Base class for LLM clients (Ollama, HuggingFace, etc.)"""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text from the given prompt"""
        pass

    @abstractmethod
    def generate_with_context(
        self,
        query: str,
        context: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text with RAG context"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM client is available"""
        pass
