#!/usr/bin/env python3
"""CLI script to collect transformer hidden states and build the critic dataset."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from services.core.domain_registry import get_domain_config
from services.critic.collector import HiddenStateCollector
from retrieval.hybrid import load_hybrid_retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect transformer hidden states for critic training.")
    parser.add_argument("--domain", required=True, help="Domain ID (e.g. medical_prescription)")
    parser.add_argument("--model-path", required=True, help="Path to Hugging Face model or merged SFT/DPO weights")
    parser.add_argument("--queries-file", help="Path to input queries JSONL file (defaults to domain evaluation dataset)")
    parser.add_argument("--output-file", help="Path to save collected hidden states (JSONL)")
    parser.add_argument("--layers", help="Comma-separated layer indices to extract (e.g., '20,24,28,32')")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max tokens to generate per query")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load base model in 4-bit for memory savings")
    args = parser.parse_args()

    # Load domain config
    domain_cfg = get_domain_config(args.domain)
    
    # Setup queries file path
    queries_path = Path(args.queries_file) if args.queries_file else domain_cfg.root / "data" / "evaluation" / args.domain / "rag_eval.jsonl"
    if not queries_path.exists():
        logger.error(f"Queries file not found at {queries_path}")
        sys.exit(1)

    # Setup output path
    output_path = Path(args.output_file) if args.output_file else domain_cfg.root / "data" / "processed" / args.domain / "critic_dataset.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Parse layers
    layer_indices = None
    if args.layers:
        try:
            layer_indices = [int(l.strip()) for l in args.layers.split(",") if l.strip()]
        except ValueError:
            logger.error("Invalid layers argument. Must be comma-separated integers.")
            sys.exit(1)

    # Load tokenizer and model
    logger.info(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Quantization configurations
    model_kwargs = {"device_map": "auto", "trust_remote_code": True}
    if args.load_in_4bit and device == "cuda":
        try:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            logger.info("Using 4-bit quantization.")
        except ImportError:
            logger.warning("bitsandbytes not available. Loading model in default precision.")

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    
    # Load collector and retriever
    collector = HiddenStateCollector(model=model, tokenizer=tokenizer, device=device)
    retriever = load_hybrid_retriever(args.domain)

    # Read queries
    logger.info(f"Reading queries from {queries_path}...")
    queries = []
    with queries_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    # Run collection loop
    logger.info("Starting collection...")
    all_records = []
    
    for idx, case in enumerate(queries, 1):
        query_text = case.get("query", "")
        if not query_text:
            continue
            
        logger.info(f"[{idx}/{len(queries)}] Processing query: {query_text[:50]}...")
        
        # Retrieve context evidence
        retrieved = retriever.search(query_text, top_k=1)
        if not retrieved:
            logger.warning(f"No evidence retrieved for query: {query_text}. Skipping.")
            continue
            
        source_chunk = retrieved[0].chunk["text"]
        source_id = retrieved[0].chunk["source_id"]

        try:
            records = collector.collect_from_query(
                query=query_text,
                source_chunk=source_chunk,
                source_id=source_id,
                layer_indices=layer_indices,
                max_new_tokens=args.max_new_tokens,
            )
            all_records.extend(records)
            logger.info(f"Collected {len(records)} layer-token state records for query.")
        except Exception as e:
            logger.error(f"Failed to collect states for query '{query_text}': {e}", exc_info=True)

    # Save output dataset
    logger.info(f"Saving {len(all_records)} total records to {output_path}...")
    with output_path.open("w", encoding="utf-8") as out:
        for record in all_records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Hidden state collection completed successfully.")


if __name__ == "__main__":
    main()
