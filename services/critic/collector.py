from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


class GroundednessLabeller:
    """Label text segments as grounded or unsupported/hallucinated against evidence sources."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def clean_tokens(self, text: str) -> set[str]:
        """Normalize and tokenize text to a set of words."""
        return {t.lower() for t in re.findall(r"[A-Za-z0-9_]+", text) if len(t) > 1}

    def label_sentences(self, answer: str, source_text: str) -> list[tuple[str, int]]:
        """Split answer into sentences and label each as grounded (1) or hallucinated (0)."""
        # Split on sentence boundaries
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
        source_words = self.clean_tokens(source_text)

        labeled = []
        for sentence in sentences:
            sentence_words = self.clean_tokens(sentence)
            if not sentence_words:
                labeled.append((sentence, 1))  # Default to grounded if no words
                continue

            intersection = sentence_words & source_words
            union = sentence_words | source_words
            jaccard = len(intersection) / len(union) if union else 0.0
            overlap_ratio = len(intersection) / len(sentence_words)

            # Mark grounded if Jaccard similarity or overlap ratio is sufficiently high
            is_grounded = 1 if (jaccard >= self.threshold or overlap_ratio >= 0.6) else 0
            labeled.append((sentence, is_grounded))

        return labeled


class HiddenStateCollector:
    """Collect hidden states from middle-to-late transformer layers during text generation."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.labeller = GroundednessLabeller()

    def collect_from_query(
        self,
        query: str,
        source_chunk: str,
        source_id: str,
        layer_indices: list[int] | None = None,
        max_new_tokens: int = 128,
        generation_kwargs: dict[str, Any] | None = None,
        logits_processor: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Run generation, capture hidden states, label tokens, and return critic records."""
        if torch is None:
            raise RuntimeError("PyTorch is required for HiddenStateCollector.")

        # Ensure model is in eval mode and configured correctly
        self.model.eval()
        generation_kwargs = generation_kwargs or {}

        # Format input using model's chat template or basic prompt
        prompt = f"Context: {source_chunk}\n\nQuery: {query}\n\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = int(inputs.input_ids.shape[1])

        # Run forward pass through generate with output_hidden_states enabled
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                output_hidden_states=True,
                return_dict_in_generate=True,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                logits_processor=logits_processor,
                **generation_kwargs,
            )

        generated_ids = outputs.sequences[0][prompt_len:]
        if len(generated_ids) == 0:
            return []

        # Reconstruct full response to align groundedness labels
        full_response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        labeled_sentences = self.labeller.label_sentences(full_response, source_chunk)

        # Align sentence spans to offsets in full_response
        sentence_spans = []
        current_pos = 0
        for sentence, label in labeled_sentences:
            idx = full_response.find(sentence, current_pos)
            if idx != -1:
                sentence_spans.append((idx, idx + len(sentence), label))
                current_pos = idx + len(sentence)
            else:
                sentence_spans.append((current_pos, current_pos + len(sentence), label))
                current_pos += len(sentence)

        # Map generated tokens to character spans in full_response
        accumulated_text = ""
        token_spans = []
        for i, token_id in enumerate(generated_ids):
            token_str = self.tokenizer.decode([token_id])
            start_idx = len(accumulated_text)
            accumulated_text += token_str
            end_idx = len(accumulated_text)
            token_spans.append((i, start_idx, end_idx, token_str))

        # Determine target layer indices
        num_layers = getattr(self.model.config, "num_hidden_layers", len(outputs.hidden_states[0]) - 1)
        if not layer_indices:
            # Default to the last 25% of layers (middle-to-late)
            start_layer = int(num_layers * 0.75)
            layer_indices = list(range(start_layer, num_layers + 1))

        # Extract tokens and hidden states
        records = []
        for step_idx, start, end, token_str in token_spans:
            # Skip empty tokens or formatting tokens
            if not token_str.strip():
                continue

            # Determine grounded label from sentence overlap span
            grounded_label = 1
            for s_start, s_end, label in sentence_spans:
                if start >= s_start and start < s_end:
                    grounded_label = label
                    break

            # Collect states for each target layer
            for layer_idx in layer_indices:
                if layer_idx < 0 or layer_idx >= len(outputs.hidden_states[step_idx]):
                    continue

                # Hidden state tensor selection handling KV caching offsets
                # step_idx 0 includes full prompt representation, so take prompt_len - 1
                # step_idx > 0 represents single-token passes, so take index 0
                step_hidden = outputs.hidden_states[step_idx][layer_idx]
                if step_idx == 0:
                    token_vector = step_hidden[0, prompt_len - 1, :]
                else:
                    token_vector = step_hidden[0, 0, :]

                records.append(
                    {
                        "token": token_str.strip(),
                        "hidden_state": token_vector.detach().cpu().tolist(),
                        "layer_index": layer_idx,
                        "source_chunk": source_chunk,
                        "source_id": source_id,
                        "grounded_label": grounded_label,
                    }
                )

        return records
