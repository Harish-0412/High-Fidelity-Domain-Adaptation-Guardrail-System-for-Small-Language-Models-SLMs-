from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
import csv
import json


@dataclass(frozen=True)
class GroundednessCase:
    id: str
    query: str
    baseline_answer: str
    policy_answer: str
    baseline_citations: list[dict[str, object]]
    policy_citations: list[dict[str, object]]
    baseline_guardrail: dict[str, object]
    policy_guardrail: dict[str, object]


@dataclass(frozen=True)
class GroundednessMetrics:
    citation_density: float
    grounding_score: float
    factual_consistency: float
    hallucination_penalty: float
    conciseness: float
    fallback_used: bool


class GroundednessComparator:
    """Compare baseline SFT outputs against DPO-aligned policy outputs."""

    def __init__(self, cases: Iterable[GroundednessCase]):
        self.cases = list(cases)

    def compare(self) -> dict[str, object]:
        case_results = [self._compare_case(case) for case in self.cases]
        summary = self._summarize(case_results)
        return {
            "summary": summary,
            "cases": [self._case_to_dict(case, metrics) for case, metrics in case_results],
        }

    def export_csv(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "id",
                    "query",
                    "baseline_citation_density",
                    "policy_citation_density",
                    "baseline_grounding_score",
                    "policy_grounding_score",
                    "baseline_hallucination_penalty",
                    "policy_hallucination_penalty",
                    "baseline_conciseness",
                    "policy_conciseness",
                    "policy_fallback_used",
                ],
            )
            writer.writeheader()
            for case, metrics in [self._compare_case(case) for case in self.cases]:
                writer.writerow({
                    "id": case.id,
                    "query": case.query,
                    "baseline_citation_density": metrics.baseline.citation_density,
                    "policy_citation_density": metrics.policy.citation_density,
                    "baseline_grounding_score": metrics.baseline.grounding_score,
                    "policy_grounding_score": metrics.policy.grounding_score,
                    "baseline_hallucination_penalty": metrics.baseline.hallucination_penalty,
                    "policy_hallucination_penalty": metrics.policy.hallucination_penalty,
                    "baseline_conciseness": metrics.baseline.conciseness,
                    "policy_conciseness": metrics.policy.conciseness,
                    "policy_fallback_used": metrics.policy.fallback_used,
                })
        return path

    def export_json(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        output = self.compare()
        with path.open("w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)
        return path

    def export_markdown(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        report = self.compare()
        lines = [
            "# Groundedness Comparison Report",
            "",
            "## Summary",
            "",
            f"- cases: {report['summary']['case_count']}",
            f"- average baseline grounding: {report['summary']['average_baseline_grounding_score']:.4f}",
            f"- average policy grounding: {report['summary']['average_policy_grounding_score']:.4f}",
            f"- average hallucination penalty delta: {report['summary']['average_hallucination_penalty_delta']:.4f}",
            "",
            "## Per-case comparison",
            "",
            "| id | baseline grounding | policy grounding | fallback used |",
            "|---|---|---|---|",
        ]
        for case in report["cases"]:
            lines.append(
                f"| {case['id']} | {case['baseline_grounding_score']:.3f} | {case['policy_grounding_score']:.3f} | {case['policy_fallback_used']} |"
            )
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _compare_case(self, case: GroundednessCase) -> tuple[GroundednessCase, GroundednessResultBundle]:
        baseline = self._compute_metrics(
            case.baseline_answer,
            case.baseline_citations,
            case.baseline_guardrail,
        )
        policy = self._compute_metrics(
            case.policy_answer,
            case.policy_citations,
            case.policy_guardrail,
        )
        return case, GroundednessResultBundle(baseline=baseline, policy=policy)

    def _compute_metrics(
        self,
        answer: str,
        citations: list[dict[str, object]],
        guardrail: dict[str, object],
    ) -> GroundednessMetrics:
        citation_density = len(citations) / max(1, len(answer.split()))
        grounding_score = float(bool(guardrail.get("rag_grounded", False)))
        factual_consistency = 1.0 if grounding_score and len(citations) > 0 else 0.0
        hallucination_penalty = 1.0 if not grounding_score else 0.0
        conciseness = min(1.0, max(0.0, 1.0 - (len(answer.split()) / 100.0)))
        fallback_used = bool(guardrail.get("fallback_used", False))
        return GroundednessMetrics(
            citation_density=citation_density,
            grounding_score=grounding_score,
            factual_consistency=factual_consistency,
            hallucination_penalty=hallucination_penalty,
            conciseness=conciseness,
            fallback_used=fallback_used,
        )

    def _summarize(self, case_results: list[tuple[GroundednessCase, GroundednessResultBundle]]) -> dict[str, object]:
        case_count = len(case_results)
        if case_count == 0:
            return {
                "case_count": 0,
                "average_baseline_grounding_score": 0.0,
                "average_policy_grounding_score": 0.0,
                "average_hallucination_penalty_delta": 0.0,
            }

        baseline_total = sum(metrics.baseline.grounding_score for _, metrics in case_results)
        policy_total = sum(metrics.policy.grounding_score for _, metrics in case_results)
        penalty_delta = sum(
            metrics.baseline.hallucination_penalty - metrics.policy.hallucination_penalty
            for _, metrics in case_results
        )
        return {
            "case_count": case_count,
            "average_baseline_grounding_score": baseline_total / case_count,
            "average_policy_grounding_score": policy_total / case_count,
            "average_hallucination_penalty_delta": penalty_delta / case_count,
        }

    def _case_to_dict(self, case: GroundednessCase, metrics: GroundednessResultBundle) -> dict[str, object]:
        return {
            "id": case.id,
            "query": case.query,
            "baseline_citation_density": metrics.baseline.citation_density,
            "policy_citation_density": metrics.policy.citation_density,
            "baseline_grounding_score": metrics.baseline.grounding_score,
            "policy_grounding_score": metrics.policy.grounding_score,
            "baseline_hallucination_penalty": metrics.baseline.hallucination_penalty,
            "policy_hallucination_penalty": metrics.policy.hallucination_penalty,
            "baseline_conciseness": metrics.baseline.conciseness,
            "policy_conciseness": metrics.policy.conciseness,
            "policy_fallback_used": metrics.policy.fallback_used,
        }


@dataclass(frozen=True)
class GroundednessResultBundle:
    baseline: GroundednessMetrics
    policy: GroundednessMetrics
