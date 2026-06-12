"""SFT Dataset builder: creates supervised fine-tuning datasets from domain chunks and general data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal
import json
import random


@dataclass(frozen=True)
class SFTExample:
    """A supervised fine-tuning example with query and ground-truth answer."""

    id: str
    query: str
    answer: str
    citations: list[dict[str, str]] = None
    source_type: Literal["domain", "general"] = "domain"
    metadata: dict[str, str] = None

    def __post_init__(self):
        if self.citations is None:
            object.__setattr__(self, "citations", [])
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "query": self.query,
            "answer": self.answer,
            "citations": self.citations,
            "source_type": self.source_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SFTExample:
        return cls(
            id=data["id"],
            query=data["query"],
            answer=data["answer"],
            citations=data.get("citations", []),
            source_type=data.get("source_type", "domain"),
            metadata=data.get("metadata", {}),
        )


class SFTDatasetBuilder:
    """Build SFT dataset from domain chunks and optional general corpus."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

    def create_from_chunks(
        self,
        chunks: Iterable[dict[str, object]],
        templates: list[str] | None = None,
    ) -> list[SFTExample]:
        """
        Generate SFT examples from domain chunks.
        
        Each chunk becomes a potential answer, and we generate queries
        from templates and chunk content.
        
        Args:
            chunks: Iterable of chunk dicts with 'text', 'chunk_id', 'source_id' fields.
            templates: Optional list of query templates like "Explain {entity}." or "What is {entity}?"
        
        Returns:
            List of SFT examples.
        """
        templates = templates or self._default_templates()
        examples: list[SFTExample] = []
        
        for chunk in chunks:
            chunk_dict = dict(chunk) if not isinstance(chunk, dict) else chunk
            text = chunk_dict.get("text", "")
            chunk_id = chunk_dict.get("chunk_id", "")
            source_id = chunk_dict.get("source_id", "")
            
            if not text.strip():
                continue
            
            # Extract key terms from the chunk text
            key_terms = self._extract_key_terms(text)
            
            # Generate queries from templates and key terms
            for template in templates:
                for term in key_terms:
                    query = template.format(entity=term)
                    example_id = f"{source_id}_{chunk_id}_{len(examples)}"
                    
                    examples.append(
                        SFTExample(
                            id=example_id,
                            query=query,
                            answer=text,
                            citations=[{"source_id": source_id, "chunk_id": chunk_id, "text": text[:200]}],
                            source_type="domain",
                            metadata={"template": template, "extracted_term": term},
                        )
                    )
        
        return examples

    def create_from_rag_examples(
        self,
        rag_results: Iterable[dict[str, object]],
    ) -> list[SFTExample]:
        """
        Create SFT examples from existing RAG query-answer pairs.
        
        Args:
            rag_results: Iterable of dicts with 'query', 'answer', 'citations'.
        
        Returns:
            List of SFT examples.
        """
        examples: list[SFTExample] = []
        
        for idx, result in enumerate(rag_results):
            example_id = result.get("id", f"rag_example_{idx}")
            examples.append(
                SFTExample(
                    id=example_id,
                    query=result.get("query", ""),
                    answer=result.get("answer", ""),
                    citations=result.get("citations", []),
                    source_type="domain",
                    metadata=result.get("metadata", {}),
                )
            )
        
        return examples

    def mix_with_general_data(
        self,
        domain_examples: list[SFTExample],
        general_examples: list[SFTExample],
        general_ratio: float = 0.2,
    ) -> list[SFTExample]:
        """
        Mix domain examples with general-purpose data to reduce catastrophic forgetting.
        
        Args:
            domain_examples: Domain-specific SFT examples.
            general_examples: General-purpose SFT examples.
            general_ratio: Fraction of general examples in final dataset (e.g., 0.2 = 20%).
        
        Returns:
            Mixed list of examples.
        """
        total_target = len(domain_examples) / (1.0 - general_ratio)
        general_count = int(total_target - len(domain_examples))
        general_count = min(general_count, len(general_examples))
        
        selected_general = random.sample(general_examples, k=general_count)
        mixed = domain_examples + selected_general
        random.shuffle(mixed)
        
        return mixed

    def export_jsonl(self, examples: list[SFTExample], path: Path | str) -> Path:
        """Export SFT examples in JSONL format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with path.open("w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
        
        return path

    def import_jsonl(self, path: Path | str) -> list[SFTExample]:
        """Load SFT examples from JSONL file."""
        path = Path(path)
        examples: list[SFTExample] = []
        
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    examples.append(SFTExample.from_dict(json.loads(line)))
        
        return examples

    def split_train_val(
        self,
        examples: list[SFTExample],
        train_ratio: float = 0.9,
    ) -> tuple[list[SFTExample], list[SFTExample]]:
        """Split examples into train/val sets."""
        random.shuffle(examples)
        split_idx = int(len(examples) * train_ratio)
        return examples[:split_idx], examples[split_idx:]

    def _default_templates(self) -> list[str]:
        """Return default query generation templates."""
        return [
            "What information is available about {entity}?",
            "Explain {entity}.",
            "What are the key points about {entity}?",
            "How does {entity} relate to medical prescription?",
            "What does the policy state regarding {entity}?",
            "Describe {entity}.",
            "What are the requirements for {entity}?",
            "When is {entity} applicable?",
            "What are the exclusions for {entity}?",
        ]

    def _extract_key_terms(self, text: str, max_terms: int = 3) -> list[str]:
        """Extract key terms from text for query generation."""
        # Simple extraction: split on punctuation and common words
        words = text.split()
        stopwords = {"the", "a", "an", "and", "or", "is", "are", "in", "on", "at", "to", "of"}
        
        # Prefer longer words and proper nouns (capitalized)
        candidates = [
            w.strip(".,;:!?()\"'") for w in words
            if len(w) > 4 and w.lower() not in stopwords
        ]
        
        if not candidates:
            candidates = [w.strip(".,;:!?()\"'") for w in words if len(w) > 2]
        
        return list(set(candidates[:max_terms]))


class GeneralDataLoader:
    """Load general-purpose SFT data from standard datasets or files."""

    @staticmethod
    def load_from_file(path: Path | str) -> list[SFTExample]:
        """Load general SFT examples from a JSONL file."""
        path = Path(path)
        examples: list[SFTExample] = []
        
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    data = json.loads(line)
                    examples.append(
                        SFTExample(
                            id=data.get("id", ""),
                            query=data.get("query", ""),
                            answer=data.get("answer", ""),
                            source_type="general",
                            citations=data.get("citations", []),
                            metadata=data.get("metadata", {}),
                        )
                    )
        
        return examples

    @staticmethod
    def create_dummy_general_data(size: int = 100) -> list[SFTExample]:
        """
        Create dummy general-purpose SFT data for testing/development.
        In production, this would be from real general corpora.
        """
        topics = [
            "biology", "physics", "chemistry", "history", "geography",
            "literature", "mathematics", "technology", "medicine", "economics"
        ]
        
        examples: list[SFTExample] = []
        for i in range(size):
            topic = random.choice(topics)
            examples.append(
                SFTExample(
                    id=f"general_{i}",
                    query=f"What is {topic}?",
                    answer=f"{topic.capitalize()} is a complex field of study with many applications and principles.",
                    source_type="general",
                    citations=[],
                    metadata={"topic": topic},
                )
            )
        
        return examples
