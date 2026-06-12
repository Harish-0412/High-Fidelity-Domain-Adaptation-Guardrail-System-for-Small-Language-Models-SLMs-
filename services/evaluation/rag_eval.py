from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from statistics import mean

from api.rag import answer_query
from services.core.config import project_root


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
    return project_root() / "data" / "evaluation" / "medical_prescription" / "rag_eval.jsonl"


def evaluate_critic_on_rag(
    cases: list[RAGEvalCase],
    model: Any,
    tokenizer: Any,
    critic: Any,
    layer_index: int,
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate a trained critic model on RAG queries, returning AUC, Precision, Recall, and F1."""
    from services.critic.collector import HiddenStateCollector
    from services.critic.trainer import calculate_metrics
    import torch

    collector = HiddenStateCollector(model=model, tokenizer=tokenizer, device=device)
    
    y_true = []
    y_scores = []
    
    for case in cases:
        from retrieval.hybrid import load_hybrid_retriever
        try:
            retriever = load_hybrid_retriever(case.domain)
            retrieved = retriever.search(case.query, top_k=1)
            if not retrieved:
                continue
            source_chunk = retrieved[0].chunk["text"]
            source_id = retrieved[0].chunk["source_id"]
        except Exception:
            source_chunk = "Mock reference context"
            source_id = "mock"

        try:
            records = collector.collect_from_query(
                query=case.query,
                source_chunk=source_chunk,
                source_id=source_id,
                layer_indices=[layer_index],
            )
        except Exception:
            continue

        if not records:
            continue

        states = [r["hidden_state"] for r in records]
        labels = [r["grounded_label"] for r in records]

        # Ground truth: 1 if all tokens are grounded, else 0
        seq_true_grounded = 0 if (0 in labels) else 1
        # Target metric is hallucination detection: 1 if hallucinated, 0 if grounded
        y_true.append(1 - seq_true_grounded)

        # Run critic model prediction
        seq_tensor = torch.tensor(states, dtype=torch.float).unsqueeze(0).to(device)
        res = critic.predict_hallucination(seq_tensor)
        y_scores.append(res["hallucination_probability"])

    if not y_true:
        return {"auc": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    return calculate_metrics(y_true, y_scores)


