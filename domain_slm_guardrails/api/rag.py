from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Sequence, List, Optional

from domain_slm_guardrails.api.schemas import Citation, GuardrailStatus, QueryResponse
from domain_slm_guardrails.core.domain_registry import get_domain_config
from domain_slm_guardrails.retrieval.hybrid import HybridResult, load_hybrid_retriever
from domain_slm_guardrails.llm import RAGGenerator, LLMConfig
from domain_slm_guardrails.guardrails.guardrails_manager import GuardrailsManager, GuardrailsResult
from domain_slm_guardrails.guardrails.hallucination_detector import HallucinationDetector
from domain_slm_guardrails.guardrails.content_moderator import ContentModerator


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

# Sentence splitter — handles common medical abbreviations (e.g. "Dr.", "No.")
# without over-splitting.
ABBREVIATIONS = (
    "Dr.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Prof.",
    "vs.",
    "No.",
    "Ref.",
    "Sec.",
    "Fig.",
    "Vol.",
    "Inc.",
    "Corp.",
    "LLC.",
    "E.M.",
)
# Improved: also split on line breaks!
SEGMENT_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9\"])"
    r"|(?=\b\d+\.\s+[A-Z])"
    r"|(?=\b[A-Z]\.\s+[A-Z])"
    r"|[\r\n]+"
)

WORD_RE = re.compile(r"[A-Za-z0-9_]+")

STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "do", "for",
    "from", "had", "has", "have", "he", "her", "him", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "may", "me", "my", "no", "not",
    "of", "on", "or", "our", "out", "per", "she", "should", "so", "such",
    "than", "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "those", "to", "up", "us", "was", "we", "were", "what", "when",
    "which", "who", "will", "with", "would", "you", "your",
})

# Minimum sentence length (chars) to be considered as a candidate answer sentence
_MIN_SENTENCE_LEN = 30
# Similarity threshold for MMR diversity pruning (0–1; higher → more diverse)
_MMR_LAMBDA = 0.6


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RAGConfig:
    min_score: float = 0.01
    max_answer_sentences: int = 3        # Lowered to 3 for more concise answers
    max_citation_chars: int = 900
    # Confidence thresholds
    low_confidence_score: float = 0.05   # below → fallback
    # Answer assembly
    use_mmr: bool = False                # Maximal Marginal Relevance deduplication
    coherence_sort: bool = False         # re-sort selected sentences by source order
    use_llm_generation: bool = True     # Use LLM for answer generation


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _query_terms(query: str) -> set[str]:
    return {
        t.lower() for t in WORD_RE.findall(query)
        if t.lower() not in STOPWORDS and len(t) > 1
    }


def _query_ngrams(query: str, domain_generic_terms: set[str], min_size: int = 2, max_size: int = 4) -> set[str]:
    tokens = [
        t.lower() for t in WORD_RE.findall(query)
        if t.lower() not in STOPWORDS and len(t) > 1
    ]
    ngrams: set[str] = set()
    for size in range(min_size, max_size + 1):
        for i in range(len(tokens) - size + 1):
            ngram_tokens = tokens[i : i + size]
            if domain_generic_terms and all(token in domain_generic_terms for token in ngram_tokens):
                continue
            ngrams.add(" ".join(ngram_tokens))
    return ngrams


def _jaccard(terms_a: set[str], terms_b: set[str]) -> float:
    """Jaccard similarity between two term sets.  Returns 0 if both empty."""
    if not terms_a or not terms_b:
        return 0.0
    return len(terms_a & terms_b) / len(terms_a | terms_b)


def _split_segments(text: str) -> list[str]:
    # Split on newlines and Markdown section separators
    segments = []
    # Split on newlines, "---", and "###"
    parts = re.split(r'\n+|---|###', text)
    for part in parts:
        part = part.strip()
        if len(part) > 20:
            segments.append(part)
    return segments


# ---------------------------------------------------------------------------
# Sentence scoring & selection
# ---------------------------------------------------------------------------

@dataclass
class _SentenceCandidate:
    text: str
    terms: set[str]
    relevance: float          # query relevance score
    source_order: int         # global position across all results


def _score_sentence(
    sentence: str,
    query_terms: set[str],
    query_ngrams: set[str],
    result_score: float,
    domain_generic_terms: set[str],
) -> float:
    """
    Score a sentence for query relevance.

    Scoring components:
    * exact_match_boost — huge boost if any query terms are found exactly
    * term_overlap   — how many query terms appear in the sentence
    * phrase_hits    — how many query n-grams appear as substrings
    * term_density   — overlap normalised by sentence vocabulary size
    * retrieval_boost — small fraction of the parent chunk's hybrid score
    * length_penalty  — slight penalty for very long sentences (>400 chars)
    * use_terms_boost — boost for terms like "use", "uses", "indicated"
    """
    sentence_terms = _query_terms(sentence)
    sentence_lower = sentence.lower()

    important_terms = query_terms - domain_generic_terms
    generic_terms = query_terms & domain_generic_terms
    important_overlap = len(important_terms & sentence_terms)
    generic_overlap = len(generic_terms & sentence_terms)
    overlap = important_overlap + generic_overlap
    phrase_hits = sum(1 for ng in query_ngrams if ng in sentence_lower)
    density = overlap / max(len(sentence_terms), 1)
    length_penalty = max(0.0, (len(sentence) - 400) / 2000)
    
    # Boost exact matches for medication names (case-insensitive)
    exact_match_boost = 0.0
    for term in query_terms:
        if term.lower() in sentence_lower:
            exact_match_boost += 30.0  # HUGE boost for exact term matches
    
    # Boost sentences with "use", "uses", "indicated", "what is", and "indicat"
    use_terms_boost = 0.0
    use_terms = {"use", "uses", "used", "indicated", "indications", "what is"}
    for term in use_terms:
        if term in sentence_lower:
            use_terms_boost += 200.0  # MASSIVE boost!
    # ULTRA HUGE boost for any mention of "indicated for" (most relevant!)
    if "indicated for" in sentence_lower:
        use_terms_boost += 10000.0
    # Boost if the segment starts with the medication name
    for term in query_terms:
        if sentence_lower.strip().startswith(term.lower()):
            use_terms_boost += 5000.0
    # HUGE boost for any mention of "indicat" (covers indications, indicated)
    if "indicat" in sentence_lower:
        use_terms_boost += 500.0
            
    # CRITICAL: If no important terms match, heavily penalize
    if important_overlap == 0:
        return -1000.0

    return (
        exact_match_boost
        + use_terms_boost
        + important_overlap * 10.0
        + generic_overlap * 0.75
        + phrase_hits * 5.0
        + density * 2.0
        + result_score * 0.05  # Almost no weight on result_score
        - length_penalty
    )


def _is_boilerplate_segment(sentence: str, page_header_pattern: str | None) -> bool:
    if not page_header_pattern:
        return False
    return bool(re.match(page_header_pattern, sentence.strip()))


def _strip_page_header_prefix(sentence: str, page_header_prefix_pattern: str | None) -> str:
    if not page_header_prefix_pattern:
        return sentence.strip()
    return re.sub(page_header_prefix_pattern, "", sentence).strip()


def _mmr_select(
    candidates: list[_SentenceCandidate],
    max_sentences: int,
    lambda_: float = _MMR_LAMBDA,
) -> list[_SentenceCandidate]:
    """
    Maximal Marginal Relevance selection.

    At each step, picks the candidate that maximises:
        λ · relevance(c) − (1−λ) · max_similarity(c, selected)

    This balances relevance with diversity, preventing near-duplicate sentences
    from flooding the answer.
    """
    if not candidates:
        return []

    selected: list[_SentenceCandidate] = []
    remaining = list(candidates)

    # Always keep the strongest relevance hit. This prevents the diversity
    # term from pushing out the one sentence that directly answers the query.
    remaining.sort(key=lambda c: c.relevance, reverse=True)
    selected.append(remaining.pop(0))

    while remaining and len(selected) < max_sentences:
        best: _SentenceCandidate | None = None
        best_mmr = float("-inf")

        for cand in remaining:
            rel = cand.relevance
            if selected:
                max_sim = max(
                    _jaccard(cand.terms, s.terms) for s in selected
                )
            else:
                max_sim = 0.0
            mmr = lambda_ * rel - (1.0 - lambda_) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best = cand

        if best is None:
            break
        selected.append(best)
        remaining.remove(best)

    return selected


def _best_sentences(
    query: str,
    results: list[HybridResult],
    max_sentences: int,
    domain_generic_terms: set[str],
    page_header_pattern: str | None = None,
    page_header_prefix_pattern: str | None = None,
    use_mmr: bool = True,
    coherence_sort: bool = True,
) -> list[str]:
    """
    Extract the best answer sentences from retrieved chunks.

    1. Score every sentence across all chunks (prioritize top retrieved chunks!).
    2. Optionally apply MMR to diversify the selection.
    3. Optionally re-sort by source order for coherent reading flow.
    """
    q_terms = _query_terms(query)
    q_ngrams = _query_ngrams(query, domain_generic_terms)

    candidates: list[_SentenceCandidate] = []
    order = 0

    for idx, result in enumerate(results):
        chunk_text = str(result.chunk.get("text", ""))
        chunk_source = str(result.chunk.get("source_id", ""))
        # Boost sentences from top retrieved chunks! (LOW boost now!)
        chunk_boost = (len(results) - idx) / len(results) * 2.0
        # MASSIVE boost for common medications guide!
        if chunk_source == "common_medications_and_conditions":
            chunk_boost += 20000.0
        
        for sentence in _split_segments(chunk_text):
            sentence = _strip_page_header_prefix(sentence.strip(), page_header_prefix_pattern)
            if len(sentence) < _MIN_SENTENCE_LEN or _is_boilerplate_segment(sentence, page_header_pattern):
                order += 1
                continue
            rel = _score_sentence(sentence, q_terms, q_ngrams, result.score, domain_generic_terms) + chunk_boost
            if rel > 0:
                candidates.append(
                    _SentenceCandidate(
                        text=sentence,
                        terms=_query_terms(sentence),
                        relevance=rel,
                        source_order=order,
                    )
                )
            order += 1

    # Deduplicate by exact normalised text
    seen: set[str] = set()
    unique_candidates: list[_SentenceCandidate] = []
    for c in candidates:
        key = c.text.lower()
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    # Sort by relevance descending before selection
    unique_candidates.sort(key=lambda c: c.relevance, reverse=True)

    if use_mmr:
        selected = _mmr_select(unique_candidates, max_sentences)
    else:
        selected = unique_candidates[:max_sentences]

    # Re-sort selected sentences by their original document order so the
    # answer reads naturally rather than jumping between topics.
    if coherence_sort:
        selected.sort(key=lambda c: c.source_order)

    return [c.text for c in selected]


# ---------------------------------------------------------------------------
# Citation building
# ---------------------------------------------------------------------------

def _make_citations(results: list[HybridResult], max_chars: int) -> list[Citation]:
    citations: list[Citation] = []
    for idx, result in enumerate(results, 1):
        chunk = result.chunk
        text = str(chunk.get("text", "")).strip()
        if len(text) > max_chars:
            # Truncate at a sentence boundary if possible
            truncated = text[: max_chars - 3]
            last_period = truncated.rfind(". ")
            if last_period > max_chars // 2:
                truncated = truncated[: last_period + 1]
            text = truncated.rstrip() + "..."
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


# ---------------------------------------------------------------------------
# Query support filter
# ---------------------------------------------------------------------------

def _has_query_support(query: str, result: HybridResult) -> bool:
    """
    Accept a result if it has at least one of:
    * non-zero BM25 score (lexical match)
    * direct term overlap with the query
    """
    if result.bm25_score > 0:
        return True
    terms = _query_terms(query)
    text_terms = _query_terms(str(result.chunk.get("text", "")))
    return bool(terms & text_terms)


# ---------------------------------------------------------------------------
# Confidence estimation
# ---------------------------------------------------------------------------

def _estimate_confidence(results: list[HybridResult], sentences: list[str]) -> float:
    """
    Lightweight confidence signal (0–1) derived from:
    * Top hybrid score
    * Number of sentences extracted vs requested
    * Whether the best result has both dense and BM25 support
    """
    if not results:
        return 0.0
    top_score = results[0].score
    sentence_ratio = min(len(sentences) / 3, 1.0)  # normalise against 3 target sentences
    dual_support = 1.0 if (results[0].dense_score > 0 and results[0].bm25_score > 0) else 0.5
    return min(top_score * 0.6 + sentence_ratio * 0.25 + dual_support * 0.15, 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _expand_query(query: str) -> str:
    """Simple query expansion/typo correction for common medication names."""
    query_lower = query.lower()
    expanded_terms = []
    
    # Common medication typos
    typo_map = {
        "asprin": "aspirin",
        "tylenol": "paracetamol acetaminophen",
        "ibuprophen": "ibuprofen",
        "ibuprofren": "ibuprofen",
        "paracip": "paracetamol",
        "amox": "amoxicillin",
        "azithro": "azithromycin",
    }
    
    for typo, correct in typo_map.items():
        if typo in query_lower:
            expanded_terms.append(correct)
    
    return " ".join(expanded_terms) if expanded_terms else None


def answer_query(
    domain: str,
    query: str,
    top_k: int = 5,
    config: RAGConfig | None = None,
    expanded_query: str | None = None,
    llm_config: LLMConfig | None = None,
    use_guardrails: bool = True,
) -> QueryResponse:
    """
    Retrieve evidence and assemble a grounded answer.

    Args:
        domain:         Domain identifier (e.g. "medical_prescription").
        query:          User's natural-language question.
        top_k:          Number of chunks to retrieve.
        config:         RAGConfig overrides.
        expanded_query: Optional query expansion string passed to BM25.
        llm_config:     LLM configuration for generation.
        use_guardrails: Whether to apply content and hallucination guardrails.
    """
    started = time.perf_counter()
    config = config or RAGConfig()
    domain_cfg = get_domain_config(domain)
    retriever = load_hybrid_retriever(domain)
    
    # Initialize guardrails
    guardrails_manager = GuardrailsManager() if use_guardrails else None
    guardrails_result: Optional[GuardrailsResult] = None

    # Auto-generate expanded query if not provided
    if expanded_query is None:
        expanded_query = _expand_query(query)

    raw_results = retriever.search(query, top_k=top_k, expanded_query=expanded_query)
    results = [
        r for r in raw_results
        if r.score >= config.min_score and _has_query_support(query, r)
    ]

    # -----------------------------------------------------------------------
    # No evidence path
    # -----------------------------------------------------------------------
    if not results:
        latency_ms = (time.perf_counter() - started) * 1000
        return QueryResponse(
            domain=domain,
            query=query,
            answer="I could not verify this from the available source documents.",
            citations=[],
            guardrail_status=GuardrailStatus(
                rag_grounded=False,
                fallback_used=True,
                reason="no_retrieval_evidence",
                json_valid=True,
                critic_score=0.0,
            ),
            latency_ms=round(latency_ms, 2),
        )

    # -----------------------------------------------------------------------
    # Evidence path - Extractive or LLM-generated approach
    # -----------------------------------------------------------------------
    citations = _make_citations(results, config.max_citation_chars)
    
    if config.use_llm_generation:
        # Initialize RAG generator
        rag_generator = RAGGenerator(llm_config=llm_config)
        generation_result = rag_generator.generate_answer(
            question=query,
            citations=citations
        )
        answer = generation_result.answer
        sentences = []  # Not used for LLM generation
    else:
        # Extractive approach
        sentences = _best_sentences(
            query,
            results,
            config.max_answer_sentences,
            domain_generic_terms=domain_cfg.domain_generic_terms,
            page_header_pattern=domain_cfg.page_header_pattern,
            page_header_prefix_pattern=domain_cfg.page_header_prefix_pattern,
            use_mmr=config.use_mmr,
            coherence_sort=config.coherence_sort,
        )

        # Assemble answer: coherently ordered sentences + up to 3 inline citation refs
        answer_body = " ".join(sentences)
        top_citation_refs = " ".join(
            f"[{c.citation_id}]" for c in citations[: min(3, len(citations))]
        )
        answer = f"{answer_body} {top_citation_refs}"

    # Apply guardrails if available
    confidence = _estimate_confidence(results, sentences)
    if guardrails_manager:
        chunk_list = [r.chunk for r in results]
        guardrails_result = guardrails_manager.apply_guardrails(
            query=query,
            generated_response=answer,
            retrieved_chunks=chunk_list,
        )
        if not guardrails_result.overall_pass:
            warning_text = " ".join(guardrails_result.warnings)
            answer = f"{answer}\n\nNote: {warning_text}"
        confidence = (confidence + guardrails_result.confidence_score) / 2

    latency_ms = (time.perf_counter() - started) * 1000
    return QueryResponse(
        domain=domain,
        query=query,
        answer=answer,
        citations=citations,
        guardrail_status=GuardrailStatus(
            rag_grounded=True,
            fallback_used=False,
            reason=None,
            json_valid=True,
            critic_score=confidence,
        ),
        latency_ms=round(latency_ms, 2),
    )
