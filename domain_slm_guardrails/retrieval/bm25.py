from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import heapq
import math
import pickle
import re
from pathlib import Path


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


@dataclass
class BM25Result:
    chunk: dict[str, object]
    score: float


class BM25Index:
    def __init__(self, chunks: list[dict[str, object]], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._build_indexes()

    def _build_indexes(self) -> None:
        self.documents = [tokenize(str(chunk.get("text", ""))) for chunk in self.chunks]
        self.doc_lengths = [len(document) for document in self.documents]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.term_frequencies = [Counter(document) for document in self.documents]
        self.doc_frequencies: Counter[str] = Counter()
        self.inverted_index: dict[str, set[int]] = defaultdict(set)
        for document in self.documents:
            self.doc_frequencies.update(set(document))
        for doc_idx, document in enumerate(self.documents):
            for term in set(document):
                self.inverted_index[term].add(doc_idx)

    def search(self, query: str, top_k: int = 5) -> list[BM25Result]:
        query_terms = tokenize(query)
        candidate_ids = self._candidate_ids(query_terms)
        scored: list[tuple[float, int]] = []
        for idx in candidate_ids:
            score = self._score(query_terms, idx)
            if score > 0:
                scored.append((score, idx))
        best = heapq.nlargest(top_k, scored, key=lambda item: item[0])
        return [BM25Result(chunk=self.chunks[idx], score=score) for score, idx in best]

    def _candidate_ids(self, query_terms: list[str]) -> set[int]:
        candidates: set[int] = set()
        for term in query_terms:
            candidates.update(self.inverted_index.get(term, set()))
        return candidates

    def _score(self, query_terms: list[str], doc_idx: int) -> float:
        if not self.documents:
            return 0.0
        doc_len = self.doc_lengths[doc_idx]
        frequencies = self.term_frequencies[doc_idx]
        total_docs = len(self.documents)
        score = 0.0
        for term in query_terms:
            term_freq = frequencies.get(term, 0)
            if term_freq == 0:
                continue
            doc_freq = self.doc_frequencies.get(term, 0)
            idf = math.log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            denominator = term_freq + self.k1 * (
                1 - self.b + self.b * (doc_len / (self.avg_doc_length or 1.0))
            )
            score += idf * (term_freq * (self.k1 + 1)) / denominator
        return score

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path: str | Path) -> "BM25Index":
        with Path(path).open("rb") as handle:
            loaded = pickle.load(handle)
        if not isinstance(loaded, BM25Index):
            raise TypeError(f"Expected BM25Index in {path}")
        if not hasattr(loaded, "inverted_index"):
            loaded._build_indexes()
        return loaded


def build_bm25_index(chunks: list[dict[str, object]]) -> BM25Index:
    return BM25Index(chunks)
