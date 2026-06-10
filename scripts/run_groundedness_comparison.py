"""Run production groundedness comparison: SFT baseline vs DPO model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domain_slm_guardrails.evaluation.groundedness_comparator import (
    GroundednessCase,
    GroundednessComparator,
)


def create_sample_groundedness_cases() -> list[GroundednessCase]:
    """Create sample cases for groundedness comparison."""
    return [
        GroundednessCase(
            id="case_1",
            query="What is CPT code 99213?",
            baseline_answer="CPT 99213 is a billing code for office visits.",
            policy_answer="CPT 99213 is an established patient office visit code for 20-29 minutes of service.",
            baseline_citations=[],
            policy_citations=[
                {
                    "source_id": "cpt_manual",
                    "text": "99213: Office visit, established patient, 20-29 min",
                }
            ],
            baseline_guardrail={"rag_grounded": False, "fallback_used": False},
            policy_guardrail={
                "rag_grounded": True,
                "fallback_used": False,
                "critic_score": 0.15,
            },
        ),
        GroundednessCase(
            id="case_2",
            query="When can modifier 25 be applied?",
            baseline_answer="Modifier 25 can be used in many situations with other modifiers.",
            policy_answer="Modifier 25 indicates a significant, separately identifiable E/M service on the same day. It can be used with procedure codes when a distinct evaluation is documented.",
            baseline_citations=[],
            policy_citations=[
                {
                    "source_id": "coding_guidelines",
                    "text": "Modifier 25: Significant, separately identifiable E/M service by same physician",
                }
            ],
            baseline_guardrail={"rag_grounded": False, "fallback_used": False},
            policy_guardrail={
                "rag_grounded": True,
                "fallback_used": False,
                "critic_score": 0.12,
            },
        ),
        GroundednessCase(
            id="case_3",
            query="What documents are needed for claim submission?",
            baseline_answer="You need several documents for claim submission, but I'm not entirely sure which ones.",
            policy_answer="Claim submission requires: patient demographics, dates of service, provider NPI, procedure codes, diagnosis codes, and supporting clinical documentation justifying medical necessity.",
            baseline_citations=[],
            policy_citations=[
                {
                    "source_id": "submission_manual",
                    "text": "Required for claim submission: demographics, DOS, NPI, procedure/diagnosis codes",
                },
                {
                    "source_id": "documentation_guide",
                    "text": "Clinical documentation must support medical necessity for all billed services",
                },
            ],
            baseline_guardrail={"rag_grounded": False, "fallback_used": True},
            policy_guardrail={
                "rag_grounded": True,
                "fallback_used": False,
                "critic_score": 0.08,
            },
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline SFT vs DPO model groundedness.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/groundedness_comparison",
        help="Output directory for comparison results.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv", "markdown"],
        default="markdown",
        help="Output format.",
    )
    args = parser.parse_args()

    # Create comparison cases
    cases = create_sample_groundedness_cases()

    # Run comparison
    comparator = GroundednessComparator(cases)
    results = comparator.compare()

    # Export results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.format == "json":
        output_file = output_dir / "comparison.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    elif args.format == "csv":
        output_file = comparator.export_csv(output_dir / "comparison.csv")
    elif args.format == "markdown":
        output_file = comparator.export_markdown(output_dir / "comparison.md")

    print(f"Comparison exported to {output_file}")
    print("\n" + "=" * 80)
    print("GROUNDEDNESS COMPARISON SUMMARY")
    print("=" * 80)
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
