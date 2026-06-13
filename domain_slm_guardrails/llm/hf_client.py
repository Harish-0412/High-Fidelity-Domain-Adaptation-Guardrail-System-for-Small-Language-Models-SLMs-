from __future__ import annotations

import logging
from typing import Optional

from domain_slm_guardrails.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("transformers package not installed. HF client will not be available.")


class HFClient(BaseLLMClient):
    """HuggingFace Transformers LLM client"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B",
        device: str = "auto",
        torch_dtype: str = "float16",
    ):
        """
        Initialize HuggingFace client.

        Args:
            model_name: Model name from HuggingFace Hub
            device: Device to use ("auto", "cuda", "cpu")
            torch_dtype: Torch data type ("float16", "float32")
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "transformers package is not installed. "
                "Install it with: pip install transformers torch"
            )

        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self._model = None
        self._tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            device_map = "auto" if self.device == "auto" else None
            dtype = torch.float16 if self.torch_dtype == "float16" else torch.float32

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype if torch.cuda.is_available() else torch.float32,
                device_map=device_map if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            self._model.eval()

            if not torch.cuda.is_available():
                self._model.to(self.device)

            logger.info(f"Loaded HF model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load HF model: {e}")
            raise

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
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        try:
            inputs = self._tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=self._tokenizer.eos_token_id,
                    **kwargs
                )

            generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the response
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()

            return generated_text
        except Exception as e:
            logger.error(f"HF generation error: {e}")
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
            **kwargs: Additional parameters

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
        """Check if HF model is available."""
        return self._model is not None and self._tokenizer is not None
