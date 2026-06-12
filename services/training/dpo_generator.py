from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Literal
import json
import random

DPORejectionStrategy = Literal[
    "weakly_cited",
    "hallucinated",
    "incomplete",
    "overly_verbose",
]


@dataclass(frozen=True)
class DPOPreferencePair:
    query: str
    chosen: str
    rejected: str
    strategy: DPORejectionStrategy
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "strategy": self.strategy,
            "metadata": self.metadata,
        }


class DPOPreferenceGenerator:
    """Generate DPO preference pairs for alignment training.

    This generator creates target preference examples from supervised fine-tuning
    examples and optional retrieval evidence, then exports them in a standard
    DPO JSONL format.
    """

    def __init__(self, seed: int = 42, template: str | None = None):
        self.seed = seed
        self.template = template or "{query}\n\nAnswer:"
        random.seed(seed)

    def generate_from_sft_examples(
        self,
        examples: Iterable[dict[str, object]],
        strategies: list[DPORejectionStrategy] | None = None,
        max_rejections_per_example: int = 1,
    ) -> list[DPOPreferencePair]:
        """Create preference pairs from SFT examples.

        Args:
            examples: Iterable of dicts containing at least `query` and `chosen`.
            strategies: Optional ordered list of rejection strategies.
            max_rejections_per_example: Number of rejected variants per example.
        """
        strategies = strategies or [
            "weakly_cited",
            "hallucinated",
            "incomplete",
            "overly_verbose",
        ]
        pairs: list[DPOPreferencePair] = []

        for example in examples:
            query = str(example.get("query", ""))
            chosen = str(example.get("chosen", example.get("answer", "")))
            citations = list(example.get("citations", []))
            evidence_text = self._join_evidence_text(citations)

            for idx in range(min(max_rejections_per_example, len(strategies))):
                strategy = strategies[idx]
                rejected = self._build_rejected_answer(chosen, evidence_text, strategy)
                pairs.append(
                    DPOPreferencePair(
                        query=query,
                        chosen=chosen,
                        rejected=rejected,
                        strategy=strategy,
                        metadata={
                            "source_type": str(example.get("source_type", "sft")),
                            "example_id": str(example.get("id", "")),
                        },
                    )
                )

        return pairs

    def export_jsonl(self, pairs: Iterable[DPOPreferencePair], path: Path | str) -> Path:
        """Export preference pairs in JSONL format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for pair in pairs:
                handle.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
        return path

    def export_standard_dpo(self, pairs: Iterable[DPOPreferencePair], path: Path | str) -> Path:
        """Export preference pairs in a standard DPO format.

        Most DPO trainers expect a JSONL file with `query`, `chosen`, and `rejected`.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for pair in pairs:
                handle.write(
                    json.dumps(
                        {
                            "query": pair.query,
                            "chosen": pair.chosen,
                            "rejected": pair.rejected,
                            "metadata": pair.metadata,
                            "strategy": pair.strategy,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return path

    def _join_evidence_text(self, citations: list[dict[str, object]]) -> str:
        if not citations:
            return ""
        return " \n".join(str(citation.get("text", citation.get("source_id", ""))) for citation in citations)

    def _build_rejected_answer(
        self,
        chosen_answer: str,
        evidence_text: str,
        strategy: DPORejectionStrategy,
    ) -> str:
        if strategy in ("weakly_cited", "poorly cited"):
            return self._make_weakly_cited(chosen_answer)
        if strategy == "hallucinated":
            return self._make_hallucinated(chosen_answer)
        if strategy == "incomplete":
            return self._make_incomplete(chosen_answer)
        if strategy in ("overly_verbose", "verbose"):
            return self._make_overly_verbose(chosen_answer)
        return chosen_answer

    def _make_weakly_cited(self, text: str) -> str:
        return (
            "According to related guidelines, "
            + text[: max(0, len(text) // 2)].strip()
            + " ..."
        )

    def _make_hallucinated(self, text: str) -> str:
        truncated = text[: max(0, len(text) // 2)].rstrip(". ")
        return f"{truncated}, and it is believed that an additional service may also be required based on similar coding rules."

    def _make_incomplete(self, text: str) -> str:
        sentences = [sentence.strip() for sentence in text.split(".") if sentence.strip()]
        if len(sentences) <= 1:
            words = text.split()
            if len(words) > 3:
                return " ".join(words[:max(1, len(words) // 2)]) + "..."
            return text[:max(1, len(text) // 2)].strip() + "..."
        return ". ".join(sentences[: max(1, len(sentences) // 2)]) + "."

    def _make_overly_verbose(self, text: str) -> str:
        phrase = "This is a clear case because the policy states that "
        return f"{phrase}{text} {text}"
