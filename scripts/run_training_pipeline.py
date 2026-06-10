"""Run end-to-end training pipeline: SFT + DPO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domain_slm_guardrails.training.pipeline import run_training_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run complete training pipeline: SFT + DPO.",
    )
    parser.add_argument(
        "--domain",
        default="medical_prescription",
        help="Domain ID to train on.",
    )
    parser.add_argument(
        "--base-model",
        default="meta-llama/Llama-2-8b",
        help="Base model identifier.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/training",
        help="Output directory for trained models.",
    )
    parser.add_argument(
        "--skip-sft",
        action="store_true",
        help="Skip SFT training.",
    )
    parser.add_argument(
        "--skip-dpo",
        action="store_true",
        help="Skip DPO training.",
    )
    args = parser.parse_args()

    results = run_training_pipeline(
        domain_id=args.domain,
        base_model=args.base_model,
        output_dir=args.output_dir,
        train_sft=not args.skip_sft,
        train_dpo=not args.skip_dpo,
    )

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
