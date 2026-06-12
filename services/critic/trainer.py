from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Iterable

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    Dataset = object
    DataLoader = None

logger = logging.getLogger(__name__)


class CriticDataset(Dataset):
    """Dataset class that groups token-level hidden state records into sequence tensors."""

    def __init__(self, records_path: Path | str, layer_index: int):
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required for CriticDataset.")
        self.records_path = Path(records_path)
        self.layer_index = layer_index
        self.sequences: list[tuple[torch.Tensor, int]] = []
        self._load_and_group_records()

    def _load_and_group_records(self) -> None:
        """Parse JSONL file and group consecutive entries into sequence representations."""
        if not self.records_path.exists():
            logger.warning(f"Dataset path {self.records_path} does not exist.")
            return

        raw_records = []
        with self.records_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_records.append(json.loads(line))

        # Filter by selected layer
        filtered = [r for r in raw_records if int(r.get("layer_index", -1)) == self.layer_index]
        if not filtered:
            logger.warning(f"No records found for layer_index {self.layer_index}.")
            return

        # Group consecutive records with the exact same source_chunk
        groups = []
        current_group = []
        last_chunk = None

        for record in filtered:
            chunk = record.get("source_chunk", "")
            if last_chunk is not None and chunk != last_chunk:
                if current_group:
                    groups.append(current_group)
                current_group = []
            
            current_group.append(record)
            last_chunk = chunk

        if current_group:
            groups.append(current_group)

        # Convert groups to sequence tensors and binary targets
        for group in groups:
            states = [r["hidden_state"] for r in group]
            labels = [r["grounded_label"] for r in group]

            # Sequence-level target: 0 if any token is hallucinated (label 0), else 1
            seq_label = 0 if (0 in labels) else 1
            
            # Form tensor shape: (seq_len, hidden_size)
            seq_tensor = torch.tensor(states, dtype=torch.float)
            self.sequences.append((seq_tensor, seq_label))

        logger.info(f"Loaded {len(self.sequences)} sequences for layer {self.layer_index} from {self.records_path}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.sequences[idx]


def pad_collate(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate function to dynamically pad variable sequence lengths in the batch."""
    sequences, labels = zip(*batch)
    padded_seqs = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
    labels_tensor = torch.tensor(labels, dtype=torch.float).unsqueeze(1)
    return padded_seqs, labels_tensor


def calculate_auc(y_true: list[int], y_scores: list[float]) -> float:
    """Calculate Area Under the ROC Curve using Mann-Whitney U test statistic."""
    if len(set(y_true)) < 2:
        return 0.5

    paired = sorted(zip(y_scores, y_true), key=lambda x: x[0])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos

    ranks_sum = sum(rank for rank, (_, label) in enumerate(paired, 1) if label == 1)
    u_stat = ranks_sum - (n_pos * (n_pos + 1)) / 2
    
    if n_pos * n_neg == 0:  # pragma: no cover
        return 0.5
    return float(u_stat / (n_pos * n_neg))


def calculate_metrics(y_true: list[int], y_scores: list[float], threshold: float = 0.5) -> dict[str, float]:
    """Calculate AUC, Precision, Recall, and F1 metrics."""
    auc = calculate_auc(y_true, y_scores)

    tp = fp = fn = tn = 0
    for true, score in zip(y_true, y_scores):
        pred = 1 if score >= threshold else 0
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1
        elif true == 0 and pred == 0:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "auc": round(auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def train_critic_model(
    model: nn.Module,
    train_dataset: CriticDataset,
    val_dataset: CriticDataset,
    epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
) -> dict[str, Any]:
    """Execute the critic training loop, evaluating metrics per epoch."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for training.")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)

    best_auc = 0.0
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * seqs.size(0)

        # Validation loop
        model.eval()
        val_scores = []
        val_targets = []
        
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs = seqs.to(device)
                outputs = model(seqs)
                val_scores.extend(outputs.squeeze(-1).cpu().tolist())
                val_targets.extend(labels.squeeze(-1).cpu().tolist())

        epoch_loss = epoch_loss / len(train_dataset)
        metrics = calculate_metrics([int(t) for t in val_targets], val_scores)
        
        logger.info(
            f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f} - Val AUC: {metrics['auc']:.4f} - F1: {metrics['f1']:.4f}"
        )

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_metrics = {**metrics, "epoch": epoch, "loss": epoch_loss}

    return best_metrics
