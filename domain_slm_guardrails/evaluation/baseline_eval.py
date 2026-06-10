"""Baseline Model Evaluation: Compare SFT and DPO models against RAG baseline."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional
import json
import logging
import time

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelEvalCase:
    """A case for model evaluation."""

    id: str
    query: str
    expected_answer: str
    expected_terms: list[str]
    expected_sources: list[str]


@dataclass(frozen=True)
class ModelEvalResult:
    """Result of evaluating a model on a single case."""

    case_id: str
    model_name: str
    answer: str
    latency_ms: float
    term_match: bool
    factual_consistency: float
    hallucination_score: float
    conciseness_score: float
    citation_score: float
    guardrail_active: bool
    passed: bool


class BaselineModelEvaluator:
    """Evaluate baseline SFT model performance against eval cases."""

    def __init__(
        self,
        model_name: str,
        model_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        """Load the model for evaluation."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading model: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model.eval()

            logger.info(f"Model loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def evaluate_cases(
        self,
        cases: Iterable[ModelEvalCase],
        max_new_tokens: int = 256,
    ) -> list[ModelEvalResult]:
        """Evaluate model on a set of cases."""
        if self.model is None:
            self.load_model()

        results: list[ModelEvalResult] = []

        for case in cases:
            result = self._evaluate_single(case, max_new_tokens)
            results.append(result)

        return results

    def _evaluate_single(
        self,
        case: ModelEvalCase,
        max_new_tokens: int,
    ) -> ModelEvalResult:
        """Evaluate model on a single case."""
        try:
            import torch

            # Prepare input
            prompt = f"Query: {case.query}\n\nAnswer:"
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.model.device)

            # Generate answer
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            latency_ms = (time.time() - start_time) * 1000

            # Decode answer
            answer = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            )
            answer = answer.replace(prompt, "").strip()

            # Compute metrics
            term_match = any(
                term.lower() in answer.lower()
                for term in case.expected_terms
            )

            factual_consistency = float(term_match)
            hallucination_score = self._compute_hallucination_score(answer, case.expected_answer)
            conciseness_score = self._compute_conciseness_score(answer)
            citation_score = 0.0  # No citations in direct model output

            passed = term_match and hallucination_score < 0.5

            return ModelEvalResult(
                case_id=case.id,
                model_name=self.model_name,
                answer=answer,
                latency_ms=latency_ms,
                term_match=term_match,
                factual_consistency=factual_consistency,
                hallucination_score=hallucination_score,
                conciseness_score=conciseness_score,
                citation_score=citation_score,
                guardrail_active=False,
                passed=passed,
            )

        except Exception as e:
            logger.error(f"Error evaluating case {case.id}: {e}")
            return ModelEvalResult(
                case_id=case.id,
                model_name=self.model_name,
                answer="ERROR",
                latency_ms=0.0,
                term_match=False,
                factual_consistency=0.0,
                hallucination_score=1.0,
                conciseness_score=0.0,
                citation_score=0.0,
                guardrail_active=False,
                passed=False,
            )

    def _compute_hallucination_score(self, answer: str, expected: str) -> float:
        """
        Estimate hallucination score (0=no hallucination, 1=hallucinated).
        
        Simplified: based on overlap with expected answer.
        """
        answer_words = set(answer.lower().split())
        expected_words = set(expected.lower().split())

        if not expected_words:
            return 0.5

        overlap = len(answer_words & expected_words)
        overlap_ratio = overlap / len(expected_words)

        # High hallucination if low overlap
        return max(0.0, 1.0 - overlap_ratio)

    def _compute_conciseness_score(self, answer: str) -> float:
        """Score conciseness (0=verbose, 1=concise)."""
        word_count = len(answer.split())
        
        if word_count < 50:
            return 1.0
        elif word_count < 200:
            return 0.8
        elif word_count < 500:
            return 0.5
        else:
            return 0.2

    def summarize_results(
        self,
        results: list[ModelEvalResult],
    ) -> dict[str, object]:
        """Summarize evaluation results."""
        if not results:
            return {"total_cases": 0, "pass_rate": 0.0}

        passed = sum(1 for r in results if r.passed)
        avg_factual = sum(r.factual_consistency for r in results) / len(results)
        avg_hallucination = sum(r.hallucination_score for r in results) / len(results)
        avg_conciseness = sum(r.conciseness_score for r in results) / len(results)
        avg_latency = sum(r.latency_ms for r in results) / len(results)

        return {
            "model_name": self.model_name,
            "total_cases": len(results),
            "passed": passed,
            "pass_rate": round(passed / len(results), 4),
            "avg_factual_consistency": round(avg_factual, 4),
            "avg_hallucination_score": round(avg_hallucination, 4),
            "avg_conciseness_score": round(avg_conciseness, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "results": [asdict(r) for r in results],
        }

    def export_results(
        self,
        results: list[ModelEvalResult],
        path: Path | str,
        format: str = "json",
    ) -> Path:
        """Export evaluation results."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            summary = self.summarize_results(results)
            with path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

        elif format == "jsonl":
            with path.open("w", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

        logger.info(f"Results exported to {path}")
        return path


class MultiModelComparison:
    """Compare performance across multiple models."""

    def __init__(self, models: dict[str, str]):
        """
        Args:
            models: Dict mapping model names to model paths/identifiers.
        """
        self.models = models
        self.evaluators = {
            name: BaselineModelEvaluator(name, path)
            for name, path in models.items()
        }

    def evaluate_all(
        self,
        cases: list[ModelEvalCase],
        max_new_tokens: int = 256,
    ) -> dict[str, list[ModelEvalResult]]:
        """Evaluate all models on the same cases."""
        results = {}

        for model_name, evaluator in self.evaluators.items():
            logger.info(f"Evaluating {model_name}...")
            try:
                results[model_name] = evaluator.evaluate_cases(cases, max_new_tokens)
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                results[model_name] = []

        return results

    def compare_summary(
        self,
        all_results: dict[str, list[ModelEvalResult]],
    ) -> dict[str, object]:
        """Generate comparison summary across models."""
        comparison = {}

        for model_name, results in all_results.items():
            evaluator = self.evaluators[model_name]
            summary = evaluator.summarize_results(results)
            comparison[model_name] = summary

        # Rank models by pass rate
        ranked = sorted(
            comparison.items(),
            key=lambda x: x[1].get("pass_rate", 0.0),
            reverse=True,
        )

        return {
            "comparison": dict(ranked),
            "best_model": ranked[0][0] if ranked else None,
            "best_pass_rate": ranked[0][1].get("pass_rate", 0.0) if ranked else 0.0,
        }

    def export_comparison(
        self,
        all_results: dict[str, list[ModelEvalResult]],
        output_dir: Path | str,
    ) -> Path:
        """Export full comparison results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export per-model results
        for model_name, results in all_results.items():
            evaluator = self.evaluators[model_name]
            result_file = output_dir / f"{model_name}_results.json"
            evaluator.export_results(results, result_file, format="json")

        # Export comparison summary
        comparison = self.compare_summary(all_results)
        summary_file = output_dir / "comparison_summary.json"
        with summary_file.open("w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)

        logger.info(f"Comparison results exported to {output_dir}")
        return output_dir
