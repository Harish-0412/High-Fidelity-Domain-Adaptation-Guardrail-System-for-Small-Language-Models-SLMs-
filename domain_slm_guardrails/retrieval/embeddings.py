from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


# ---------------------------------------------------------------------------
# Shared protocol so the rest of the codebase is not tied to a concrete class
# ---------------------------------------------------------------------------

@runtime_checkable
class EmbeddingModel(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


# ---------------------------------------------------------------------------
# Hash-based fallback model
# ---------------------------------------------------------------------------

@dataclass
class HashingEmbeddingModel:
    """
    Deterministic hashing embedding — zero external dependencies.

    Improvements over original:
    * Uses two independent hash families (blake2b + sha256) per token so
      collisions are far less correlated, increasing semantic discrimination.
    * Bigrams are also hashed and added at half weight, giving the vector a
      weak phrase-level signal.
    * L2 normalisation is done once per text (unchanged) but the norm guard
      uses a small epsilon instead of zero to avoid division artefacts.
    """

    dimension: int = 384

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> list[float]:
        tokens = _tokens(text)
        vector = [0.0] * self.dimension

        # Unigrams — two independent hash projections
        for token in tokens:
            raw = token.encode("utf-8")
            self._project(raw, vector, weight=1.0)

        # Bigrams — weaker phrase signal
        for a, b in zip(tokens, tokens[1:]):
            raw = f"{a} {b}".encode("utf-8")
            self._project(raw, vector, weight=0.5)

        return _l2_normalize(vector)

    def _project(self, raw: bytes, vector: list[float], weight: float) -> None:
        # First hash family: blake2b
        d1 = hashlib.blake2b(raw, digest_size=8).digest()
        bucket1 = int.from_bytes(d1[:4], "big") % self.dimension
        sign1 = 1.0 if d1[4] % 2 == 0 else -1.0
        vector[bucket1] += sign1 * weight

        # Second hash family: sha256 (independent collision pattern)
        d2 = hashlib.sha256(raw).digest()
        bucket2 = int.from_bytes(d2[:4], "big") % self.dimension
        sign2 = 1.0 if d2[4] % 2 == 0 else -1.0
        vector[bucket2] += sign2 * weight * 0.7   # downweight to avoid dominating


# ---------------------------------------------------------------------------
# Sentence-transformer wrapper
# ---------------------------------------------------------------------------

@dataclass
class SentenceTransformerEmbeddingModel:
    model_name: str
    batch_size: int = 64      # exposed so callers can tune for GPU/CPU memory

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode in batches; returns L2-normalised float lists."""
        result: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vectors = self.model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            result.extend(v.tolist() for v in vectors)
        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_embedding_model(model_name: str, dimension: int = 384) -> EmbeddingModel:
    """
    Returns a SentenceTransformerEmbeddingModel when the library is available,
    otherwise falls back to HashingEmbeddingModel (no extra deps required).
    """
    if model_name == "local-hashing":
        return HashingEmbeddingModel(dimension=dimension)
    try:
        return SentenceTransformerEmbeddingModel(model_name=model_name)
    except ModuleNotFoundError:
        return HashingEmbeddingModel(dimension=dimension)


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm < 1e-10:
        return vector
    inv = 1.0 / norm
    return [v * inv for v in vector]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Dot product of two already-normalised vectors; falls back to full calc."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimension")
    dot = sum(x * y for x, y in zip(a, b))
    # If vectors were pre-normalised the denominator ≈ 1 — check anyway
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    denom = norm_a * norm_b
    return dot / denom if denom > 1e-10 else 0.0