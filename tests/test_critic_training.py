from __future__ import annotations

import tempfile
import pytest
from pathlib import Path

# Skip test module if torch is not installed
torch = pytest.importorskip("torch")

import torch.nn as nn
from services.critic.models import BiLSTMCritic, CNNCritic
from services.critic.trainer import (
    CriticDataset,
    calculate_auc,
    calculate_metrics,
    train_critic_model,
)


# ---------------------------------------------------------------------------
# Setup Synthetic Datasets
# ---------------------------------------------------------------------------

class DummyDataset(torch.utils.data.Dataset):
    """Synthetic dataset generating distinct random sequences for test convergence."""

    def __init__(self, size=20, seq_len=5, hidden_size=8, seed=42):
        torch.manual_seed(seed)
        self.sequences = []
        for i in range(size):
            # Positives (grounded=1) have higher values than negatives (grounded=0/hallucinated)
            label = i % 2
            val = 2.0 if label == 1 else -2.0
            seq_tensor = torch.randn((seq_len, hidden_size)) + val
            self.sequences.append((seq_tensor, label))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.sequences[idx]


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_metrics_calculation_auc():
    # Simple deterministic test cases for Mann-Whitney AUC
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.35, 0.8]
    
    # Positive ranks (sorted ascending):
    # 0.1 (true=0) -> rank 1
    # 0.35 (true=1) -> rank 2
    # 0.4 (true=0) -> rank 3
    # 0.8 (true=1) -> rank 4
    # rank sum = 2 + 4 = 6
    # U-stat = 6 - (2*3)/2 = 3
    # AUC = 3 / (2*2) = 0.75
    auc = calculate_auc(y_true, y_scores)
    assert auc == 0.75

    # Test complete metrics dict
    metrics = calculate_metrics(y_true, y_scores, threshold=0.5)
    assert metrics["auc"] == 0.75
    # Predictions at threshold 0.5: [0, 0, 0, 1]
    # TP: 1 (score 0.8), FP: 0, FN: 1 (score 0.35), TN: 2
    # Precision: 1 / 1 = 1.0
    # Recall: 1 / 2 = 0.5
    # F1: 2 * (1 * 0.5) / 1.5 = 2/3 = 0.6667
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.6667


def test_bilstm_critic_forward_shape():
    model = BiLSTMCritic(hidden_size=16, lstm_hidden=8, num_layers=1)
    
    # Input batch of size 2, sequence length 5, hidden size 16
    x = torch.randn((2, 5, 16))
    output = model(x)
    
    assert output.shape == (2, 1)
    assert all(0.0 <= val <= 1.0 for val in output.squeeze(-1).tolist())


def test_bilstm_critic_predict_hallucination():
    model = BiLSTMCritic(hidden_size=8, lstm_hidden=4)
    x = torch.randn((1, 5, 8))
    
    res = model.predict_hallucination(x)
    
    assert "hallucination_probability" in res
    assert "confidence_score" in res
    assert 0.0 <= res["hallucination_probability"] <= 1.0
    assert 0.0 <= res["confidence_score"] <= 1.0


def test_cnn_critic_forward_shape():
    model = CNNCritic(hidden_size=16, cnn_channels=8, kernel_size=3)
    
    # Input batch of size 2, sequence length 5, hidden size 16
    x = torch.randn((2, 5, 16))
    output = model(x)
    
    assert output.shape == (2, 1)
    assert all(0.0 <= val <= 1.0 for val in output.squeeze(-1).tolist())


def test_critic_model_training_loop():
    model = BiLSTMCritic(hidden_size=8, lstm_hidden=4)
    
    train_dataset = DummyDataset(size=16, seq_len=4, hidden_size=8)
    val_dataset = DummyDataset(size=4, seq_len=4, hidden_size=8)
    
    metrics = train_critic_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=3,
        batch_size=4,
        learning_rate=1e-2,
        device="cpu",
    )
    
    assert "auc" in metrics
    assert "f1" in metrics
    assert "epoch" in metrics
    assert metrics["epoch"] >= 1
    assert 0.0 <= metrics["auc"] <= 1.0
