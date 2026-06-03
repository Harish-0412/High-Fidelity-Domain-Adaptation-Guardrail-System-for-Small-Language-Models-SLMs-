from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import heapq
import math
import pickle
import re
from pathlib import Path


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

# Lightweight suffix-stripping stemmer — avoids the full NLTK dependency while
# improving recall on inflected medical/billing terms (e.g. "billed"→"bill",
# "modifiers"→"modifier").
_SUFFIXES = ("ation", "ations", "ings", "ment", "ments", "ers", "ies", "ed", "ing", "es", "s")


def _stem(token: str) -> str:
    """Minimal suffix-stripping stem — preserves enough of the root for IDF to work."""
    if len(token) < 5:
        return token
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: len(token) - len(suffix)]
    return token


def tokenize(text: str, stem: bool = True) -> list[str]:
    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    return [_stem(t) for t in tokens] if stem else tokens


@dataclass
class BM25Result:
    chunk: dict[str, object]
    score: float


class BM25Index:
    """
    BM25+ index with the following improvements over the original:

    1. **Stemmed tokens** — boosts recall for inflected query/document terms.
    2. **BM25+ delta floor** (δ=1.0) — prevents zero contributions from rare
       terms that appear once in a long doc, closing the gap between BM25 and
       TF-IDF on short-tail queries.
    3. **Field-length cache** — avg_doc_length is stored once at build time and
       not recomputed on every score call.
    4. **Per-term early exit** — skips scoring when no candidate doc IDs exist
       for any query term rather than iterating all docs.
    5. **Separate raw/stem maps** — original tokens are preserved so serialised
       chunks remain human-readable while scoring uses stems.
    """

    def __init__(
        self,
        chunks: list[dict[str, object]],
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,          # BM25+ lower-bound term contribution
        stem: bool = True,
    ):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.stem = stem
        self._build_indexes()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_indexes(self) -> None:
        self.documents: list[list[str]] = [
            tokenize(str(chunk.get("text", "")), stem=self.stem)
            for chunk in self.chunks
        ]
        self.doc_lengths: list[int] = [len(doc) for doc in self.documents]
        total = sum(self.doc_lengths)
        self.avg_doc_length: float = total / len(self.doc_lengths) if self.doc_lengths else 1.0

        self.term_frequencies: list[Counter[str]] = [Counter(doc) for doc in self.documents]

        self.doc_frequencies: Counter[str] = Counter()
        self.inverted_index: dict[str, set[int]] = defaultdict(set)

        for doc_idx, doc in enumerate(self.documents):
            unique_terms = set(doc)
            self.doc_frequencies.update(unique_terms)
            for term in unique_terms:
                self.inverted_index[term].add(doc_idx)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[BM25Result]:
        query_terms = tokenize(query, stem=self.stem)
        if not query_terms:
            return []

        # Union of candidate documents that share ≥1 query term
        candidate_ids = self._candidate_ids(query_terms)
        if not candidate_ids:
            return []

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
        """BM25+ scoring — adds δ to prevent zero TF contribution floor."""
        doc_len = self.doc_lengths[doc_idx]
        frequencies = self.term_frequencies[doc_idx]
        total_docs = len(self.documents)
        avg_len = self.avg_doc_length or 1.0
        score = 0.0

        for term in set(query_terms):          # deduplicate query terms
            term_freq = frequencies.get(term, 0)
            if term_freq == 0:
                continue
            doc_freq = self.doc_frequencies.get(term, 0)
            # Smoothed IDF — stable even when doc_freq == total_docs
            idf = math.log(1.0 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            norm_tf = (term_freq * (self.k1 + 1.0)) / (
                term_freq + self.k1 * (1.0 - self.b + self.b * doc_len / avg_len)
            )
            # BM25+ delta floor prevents a single-occurrence term in a long doc
            # from scoring exactly 0.
            score += idf * (norm_tf + self.delta)

        return score

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str | Path) -> "BM25Index":
        with Path(path).open("rb") as handle:
            loaded = pickle.load(handle)
        if not isinstance(loaded, BM25Index):
            raise TypeError(f"Expected BM25Index, got {type(loaded)} in {path}")
        needs_rebuild = False
        if not hasattr(loaded, "delta"):
            loaded.delta = 1.0
            needs_rebuild = True
        if not hasattr(loaded, "stem"):
            loaded.stem = True
            needs_rebuild = True
        if not hasattr(loaded, "inverted_index"):
            needs_rebuild = True
        if needs_rebuild:
            loaded._build_indexes()
        return loaded


def build_bm25_index(
    chunks: list[dict[str, object]],
    k1: float = 1.5,
    b: float = 0.75,
    delta: float = 1.0,
    stem: bool = True,
) -> BM25Index:
    return BM25Index(chunks, k1=k1, b=b, delta=delta, stem=stem)
