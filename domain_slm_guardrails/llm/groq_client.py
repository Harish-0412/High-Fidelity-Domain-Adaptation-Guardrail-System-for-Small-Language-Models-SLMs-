
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from groq import Groq
except ImportError:
    Groq = None


@dataclass(frozen=True)
class LLMConfig:
    model: str = "llama-3.1-8b-instant"
    api_key: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.9


class GroqClient:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        if Groq is None:
            raise ImportError(
                "groq package not installed. Install with: pip install groq"
            )
        self.client = Groq(api_key=self.config.api_key)

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

        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            top_p=cfg.top_p,
        )
        return chat_completion.choices[0].message.content
