from __future__ import annotations

from dataclasses import dataclass
import re
import time

from domain_slm_guardrails.api.schemas import Citation, GuardrailStatus, QueryResponse
from domain_slm_guardrails.retrieval.hybrid import HybridResult, load_hybrid_retriever


SEGMENT_RE = re.compile(r"(?<=[.!?])\s+|(?=\b\d+\.\s+[A-Z])|(?=\b[A-Z]\.\s+[A-Z])")
WORD_RE = re.compile(r"[A-Za-z0-9_]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "for",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "the",
    "to",
    "what",
    "when",
    "with",
}


@dataclass(frozen=True)
class RAGConfig:
    min_score: float = 0.01
    max_answer_sentences: int = 4
    max_citation_chars: int = 900


def _query_terms(query: str) -> set[str]:
    return {
        token.lower()
        for token in WORD_RE.findall(query)
        if token.lower() not in STOPWORDS and len(token) > 1
    }


def _query_ngrams(query: str, min_size: int = 2, max_size: int = 4) -> set[str]:
    tokens = [
        token.lower()
        for token in WORD_RE.findall(query)
        if token.lower() not in STOPWORDS and len(token) > 1
    ]
    ngrams: set[str] = set()
    for size in range(min_size, max_size + 1):
        for idx in range(0, len(tokens) - size + 1):
            ngrams.add(" ".join(tokens[idx : idx + size]))
    return ngrams


def _best_sentences(query: str, results: list[HybridResult], max_sentences: int) -> list[str]:
    terms = _query_terms(query)
    ngrams = _query_ngrams(query)
    scored: list[tuple[float, int, str]] = []
    order = 0
    for result in results:
        for sentence in SEGMENT_RE.split(str(result.chunk.get("text", ""))):
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            sentence_terms = _query_terms(sentence)
            sentence_lower = sentence.lower()
            overlap = len(terms.intersection(sentence_terms))
            phrase_hits = sum(1 for ngram in ngrams if ngram in sentence_lower)
            density = overlap / max(len(sentence_terms), 1)
            compactness_penalty = min(len(sentence) / 900, 0.5)
            score = (overlap * 2.0) + (phrase_hits * 3.0) + density + (0.25 * result.score) - compactness_penalty
            if score > 0:
                scored.append((score, order, sentence))
            order += 1
    scored.sort(key=lambda item: (-item[0], item[1]))

    selected: list[str] = []
    seen: set[str] = set()
    for _, _, sentence in scored:
        normalized = sentence.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(sentence)
        if len(selected) >= max_sentences:
            break
    return selected


def _make_citations(results: list[HybridResult], max_chars: int) -> list[Citation]:
    citations: list[Citation] = []
    for idx, result in enumerate(results, 1):
        chunk = result.chunk
        text = str(chunk.get("text", "")).strip()
        if len(text) > max_chars:
            text = text[: max_chars - 3].rstrip() + "..."
        page = chunk.get("page")
        citations.append(
            Citation(
                citation_id=f"C{idx}",
                chunk_id=str(chunk["chunk_id"]),
                source_id=str(chunk["source_id"]),
                page=int(page) if page is not None else None,
                score=round(float(result.score), 4),
                text=text,
            )
        )
    return citations


def _has_query_support(query: str, result: HybridResult) -> bool:
    if result.bm25_score > 0:
        return True
    terms = _query_terms(query)
    text_terms = _query_terms(str(result.chunk.get("text", "")))
    return bool(terms.intersection(text_terms))


def answer_query(
    domain: str,
    query: str,
    top_k: int = 5,
    config: RAGConfig | None = None,
) -> QueryResponse:
    started = time.perf_counter()
    config = config or RAGConfig()
    retriever = load_hybrid_retriever(domain)
    results = [
        result
        for result in retriever.search(query, top_k=top_k)
        if result.score >= config.min_score and _has_query_support(query, result)
    ]

    if not results:
        answer = "I could not verify this from the available source documents."
        citations: list[Citation] = []
        guardrail = GuardrailStatus(
            rag_grounded=False,
            fallback_used=True,
            reason="no_retrieval_evidence",
        )
    else:
        citations = _make_citations(results, config.max_citation_chars)
        sentences = _best_sentences(query, results, config.max_answer_sentences)
        if sentences:
            cited = " ".join(sentences)
            citation_refs = " ".join(f"[{citation.citation_id}]" for citation in citations[: min(3, len(citations))])
            answer = f"{cited} {citation_refs}"
        else:
            answer = f"The retrieved sources contain relevant evidence for this question. [{citations[0].citation_id}]"
        guardrail = GuardrailStatus(rag_grounded=True)

    latency_ms = (time.perf_counter() - started) * 1000
    return QueryResponse(
        domain=domain,
        query=query,
        answer=answer,
        citations=citations,
        guardrail_status=guardrail,
        latency_ms=round(latency_ms, 2),
    )
