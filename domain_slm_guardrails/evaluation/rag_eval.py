from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from statistics import mean

from domain_slm_guardrails.api.rag import answer_query
from domain_slm_guardrails.core.config import project_root


@dataclass(frozen=True)
class RAGEvalCase:
    id: str
    domain: str
    query: str
    expected_terms: list[str]
    expected_source_ids: list[str]


@dataclass(frozen=True)
class RAGEvalResult:
    id: str
    passed: bool
    citation_present: bool
    expected_source_hit: bool
    expected_terms_hit: bool
    grounded: bool
    latency_ms: float
    answer: str


def load_eval_cases(path: str | Path) -> list[RAGEvalCase]:
    cases: list[RAGEvalCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            cases.append(RAGEvalCase(**raw))
    return cases


def evaluate_cases(cases: list[RAGEvalCase], top_k: int = 5) -> list[RAGEvalResult]:
    results: list[RAGEvalResult] = []
    for case in cases:
        response = answer_query(case.domain, case.query, top_k=top_k)
        answer_text = response.answer.lower()
        citation_sources = {citation.source_id for citation in response.citations}
        citation_text = " ".join(citation.text.lower() for citation in response.citations)
        citation_present = bool(response.citations)
        expected_source_hit = bool(citation_sources.intersection(case.expected_source_ids))
        expected_terms_hit = any(
            term.lower() in answer_text or term.lower() in citation_text
            for term in case.expected_terms
        )
        grounded = response.guardrail_status.rag_grounded
        passed = citation_present and expected_source_hit and expected_terms_hit and grounded
        results.append(
            RAGEvalResult(
                id=case.id,
                passed=passed,
                citation_present=citation_present,
                expected_source_hit=expected_source_hit,
                expected_terms_hit=expected_terms_hit,
                grounded=grounded,
                latency_ms=response.latency_ms,
                answer=response.answer,
            )
        )
    return results


def summarize_results(results: list[RAGEvalResult]) -> dict[str, object]:
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "avg_latency_ms": round(mean(result.latency_ms for result in results), 2) if results else 0.0,
        "results": [asdict(result) for result in results],
    }


def default_eval_path() -> Path:
    return project_root() / "data" / "evaluation" / "medical_billing" / "rag_eval.jsonl"

