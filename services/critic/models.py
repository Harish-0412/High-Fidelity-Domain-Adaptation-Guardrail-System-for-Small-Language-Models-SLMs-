from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

BaseModule = nn.Module if (nn is not None) else object


class BiLSTMCritic(BaseModule):
    """Sequence classifier using a Bidirectional LSTM to predict hallucination probability."""

    def __init__(
        self,
        hidden_size: int = 4096,  # Default for Llama-8B / Qwen-3B models
        lstm_hidden: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        if nn is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required for BiLSTMCritic.")
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_hidden = lstm_hidden
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass.
        x shape: (batch_size, seq_len, hidden_size)
        returns probability of groundedness (shape: batch_size, 1)
        """
        # LSTM output shape: (batch_size, seq_len, lstm_hidden * 2)
        lstm_out, _ = self.lstm(x)
        
        # Global average pooling over the sequence dimension
        # (Handling padding masking simply by taking mean over the seq dimension)
        pooled = torch.mean(lstm_out, dim=1)
        
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return torch.sigmoid(logits)

    def predict_hallucination(self, x: torch.Tensor) -> dict[str, float]:
        """
        Inference helper returning hallucination probability and confidence score.
        """
        self.eval()
        with torch.no_grad():
            prob_grounded = self.forward(x).item()
            
        prob_hallucination = 1.0 - prob_grounded
        confidence_score = abs(prob_grounded - 0.5) * 2.0
        
        return {
            "hallucination_probability": round(prob_hallucination, 4),
            "confidence_score": round(confidence_score, 4),
        }


class CNNCritic(BaseModule):
    """Sequence classifier using a 1D CNN to predict hallucination probability."""

    def __init__(
        self,
        hidden_size: int = 4096,
        cnn_channels: int = 128,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        if nn is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required for CNNCritic.")
        super().__init__()
        self.hidden_size = hidden_size
        self.cnn_channels = cnn_channels

        # Conv1d expects shape: (batch_size, hidden_size, seq_len)
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=cnn_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.activation = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(cnn_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x shape: (batch_size, seq_len, hidden_size)
        returns probability of groundedness (shape: batch_size, 1)
        """
        # Permute to (batch_size, hidden_size, seq_len) for Conv1d
        x_transposed = x.permute(0, 2, 1)
        
        conv_out = self.activation(self.conv(x_transposed))
        pooled = self.pool(conv_out).squeeze(-1)  # shape: (batch_size, cnn_channels)
        
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return torch.sigmoid(logits)

    def predict_hallucination(self, x: torch.Tensor) -> dict[str, float]:
        """
        Inference helper returning hallucination probability and confidence score.
        """
        self.eval()
        with torch.no_grad():
            prob_grounded = self.forward(x).item()
            
        prob_hallucination = 1.0 - prob_grounded
        confidence_score = abs(prob_grounded - 0.5) * 2.0
        
        return {
            "hallucination_probability": round(prob_hallucination, 4),
            "confidence_score": round(confidence_score, 4),
        }
