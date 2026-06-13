from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import ollama
except ImportError:
    ollama = None


@dataclass(frozen=True)
class LLMConfig:
    model: str = "llama3.2"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.9
    top_k: int = 40


class OllamaClient:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        if ollama is None:
            raise ImportError(
                "ollama package not installed. Install with: pip install ollama"
            )
        self.client = ollama.Client(host=self.config.base_url)

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        config: LLMConfig | None = None,
    ) -> str:
        cfg = config or self.config
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat(
            model=cfg.model,
            messages=messages,
            options={
                "temperature": cfg.temperature,
                "num_predict": cfg.max_tokens,
                "top_p": cfg.top_p,
                "top_k": cfg.top_k,
            },
        )
        return response["message"]["content"]
