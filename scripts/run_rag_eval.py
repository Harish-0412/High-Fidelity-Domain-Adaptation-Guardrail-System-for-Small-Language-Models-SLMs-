from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domain_slm_guardrails.evaluation.rag_eval import (  # noqa: E402
    default_eval_path,
    evaluate_cases,
    load_eval_cases,
    summarize_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the initial RAG citation evaluation set.")
    parser.add_argument("--eval-file", default=str(default_eval_path()))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    cases = load_eval_cases(args.eval_file)
    results = evaluate_cases(cases, top_k=args.top_k)
    print(json.dumps(summarize_results(results), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

