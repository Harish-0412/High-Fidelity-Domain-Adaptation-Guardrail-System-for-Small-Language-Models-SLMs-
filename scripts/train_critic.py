#!/usr/bin/env python3
"""CLI script to train the Hallucination Critic model."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import random_split

from domain_slm_guardrails.critic.models import BiLSTMCritic, CNNCritic
from domain_slm_guardrails.critic.trainer import CriticDataset, train_critic_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Hallucination Critic model.")
    parser.add_argument("--dataset-path", required=True, help="Path to collected hidden-state JSONL file")
    parser.add_argument("--model-type", choices=["bilstm", "cnn"], default="bilstm", help="Critic model backbone")
    parser.add_argument("--output-dir", default="outputs/critic", help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--layer-index", type=int, default=28, help="Layer index to extract states from")
    parser.add_argument("--hidden-size", type=int, default=4096, help="Model hidden dimension size")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on device: {device}")

    # Load dataset
    logger.info(f"Loading dataset from {args.dataset-path} for layer {args.layer_index}...")
    try:
        full_dataset = CriticDataset(args.dataset_path, layer_index=args.layer_index)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}", exc_info=True)
        sys.exit(1)

    if len(full_dataset) < 2:
        logger.error("Dataset must contain at least 2 sequences for train/validation split.")
        sys.exit(1)

    # Train/Validation split (80/20)
    val_size = max(1, int(len(full_dataset) * 0.2))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    logger.info(f"Split dataset: {train_size} training sequences, {val_size} validation sequences")

    # Initialize model
    logger.info(f"Initializing {args.model_type.upper()} model (hidden_size={args.hidden_size})...")
    if args.model_type == "bilstm":
        model = BiLSTMCritic(hidden_size=args.hidden_size, lstm_hidden=128)
    else:
        model = CNNCritic(hidden_size=args.hidden_size, cnn_channels=128)

    # Train
    logger.info("Starting training loop...")
    metrics = train_critic_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
    )

    # Save model checkpoint
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "critic_model.pt"
    
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": args.model_type,
            "hidden_size": args.hidden_size,
            "layer_index": args.layer_index,
            "metrics": metrics,
        },
        model_path,
    )
    
    logger.info(f"Saved best model checkpoint to {model_path}")
    logger.info(f"Best metrics: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f} (Epoch {metrics['epoch']})")


if __name__ == "__main__":
    main()
