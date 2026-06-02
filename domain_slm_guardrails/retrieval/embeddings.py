from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass


WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


@dataclass
class HashingEmbeddingModel:
    dimension: int = 384

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in _tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


@dataclass
class SentenceTransformerEmbeddingModel:
    model_name: str

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]


def load_embedding_model(model_name: str, dimension: int = 384):
    if model_name == "local-hashing":
        return HashingEmbeddingModel(dimension=dimension)
    try:
        return SentenceTransformerEmbeddingModel(model_name=model_name)
    except ModuleNotFoundError:
        return HashingEmbeddingModel(dimension=dimension)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimension")
    denom_a = math.sqrt(sum(value * value for value in a))
    denom_b = math.sqrt(sum(value * value for value in b))
    if denom_a == 0 or denom_b == 0:
        return 0.0
    return sum(left * right for left, right in zip(a, b)) / (denom_a * denom_b)

