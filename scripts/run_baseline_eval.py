"""Run baseline model evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domain_slm_guardrails.evaluation.baseline_eval import (
    ModelEvalCase,
    BaselineModelEvaluator,
    MultiModelComparison,
)


def create_sample_eval_cases() -> list[ModelEvalCase]:
    """Create sample evaluation cases for testing."""
    return [
        ModelEvalCase(
            id="medical_1",
            query="What is CPT code 99213?",
            expected_answer="CPT 99213 is an established patient office visit code",
            expected_terms=["CPT", "99213", "office", "visit"],
            expected_sources=["cpt_codes"],
        ),
        ModelEvalCase(
            id="medical_2",
            query="When can modifier 25 be used?",
            expected_answer="Modifier 25 indicates a significant, separately identifiable service",
            expected_terms=["modifier", "25", "service"],
            expected_sources=["coding_guidelines"],
        ),
        ModelEvalCase(
            id="medical_3",
            query="What are the requirements for claim submission?",
            expected_answer="Claims must include patient identifier, service date, and provider information",
            expected_terms=["claim", "patient", "date"],
            expected_sources=["claim_procedures"],
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline model performance.",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-2-8b",
        help="Model to evaluate.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Multiple models to compare.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/baseline_eval",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens to generate.",
    )
    args = parser.parse_args()

    # Create eval cases
    eval_cases = create_sample_eval_cases()

    if args.models:
        # Compare multiple models
        models_dict = {model: model for model in args.models}
        comparison = MultiModelComparison(models_dict)
        all_results = comparison.evaluate_all(eval_cases, max_new_tokens=args.max_tokens)
        summary = comparison.compare_summary(all_results)
        comparison.export_comparison(all_results, args.output_dir)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        # Evaluate single model
        evaluator = BaselineModelEvaluator(args.model)
        results = evaluator.evaluate_cases(eval_cases, max_new_tokens=args.max_tokens)
        summary = evaluator.summarize_results(results)
        evaluator.export_results(results, Path(args.output_dir) / "results.json")
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
