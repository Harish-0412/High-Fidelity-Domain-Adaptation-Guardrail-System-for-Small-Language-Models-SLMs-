from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from domain_slm_guardrails.core.config import project_root
from domain_slm_guardrails.core.domain_registry import get_domain_config
from domain_slm_guardrails.critic.collector import GroundednessLabeller
from domain_slm_guardrails.critic.models import BiLSTMCritic, CNNCritic

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)


class LiveGuardrailEnforcer:
    """Inference-time enforcer that scores hallucination risk and executes fallback rules."""

    def __init__(self, checkpoint_path: str | Path | None = None):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.critic_model: torch.nn.Module | None = None
        self.critic_metadata: dict[str, Any] = {}
        self.labeller = GroundednessLabeller()
        
        # Thread-safe telemetry
        self._lock = threading.Lock()
        self.metrics: dict[str, Any] = {
            "total_queries": 0,
            "total_fallbacks": 0,
            "cumulative_critic_score": 0.0,
        }
        
        # Runtime configuration overrides
        self.runtime_threshold_overrides: dict[str, float] = {}
        
        # Default audit log location
        self.audit_log_path = project_root() / "logs" / "guardrail_audit.log"
        
        if self.checkpoint_path:
            self.load_checkpoint(self.checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load a trained critic model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            logger.warning(f"Critic checkpoint path {checkpoint_path} does not exist. Critic model not loaded.")
            return

        if torch is None:
            logger.warning("PyTorch is not installed. Cannot load PyTorch Critic model checkpoint.")
            return

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.critic_metadata = {
                "model_type": checkpoint.get("model_type", "bilstm"),
                "hidden_size": checkpoint.get("hidden_size", 4096),
                "layer_index": checkpoint.get("layer_index", 28),
                "metrics": checkpoint.get("metrics", {}),
            }
            
            # Initialize appropriate model structure
            hidden_size = self.critic_metadata["hidden_size"]
            model_type = self.critic_metadata["model_type"]
            
            if model_type == "bilstm":
                self.critic_model = BiLSTMCritic(hidden_size=hidden_size, lstm_hidden=128)
            elif model_type == "cnn":
                self.critic_model = CNNCritic(hidden_size=hidden_size, cnn_channels=128)
            else:
                raise ValueError(f"Unknown critic model type: {model_type}")

            self.critic_model.load_state_dict(checkpoint["model_state_dict"])
            self.critic_model.eval()
            logger.info(f"Successfully loaded {model_type.upper()} Critic model from {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load Critic model checkpoint: {e}", exc_info=True)

    def score_sequence_tensor(self, seq_tensor: torch.Tensor) -> float:
        """Score a pre-extracted hidden-state sequence tensor directly using the Critic model."""
        if not self.critic_model:
            raise RuntimeError("Critic model is not loaded.")
        if torch is None:
            raise RuntimeError("PyTorch is required to score sequence tensors.")

        # Reshape to batch size of 1 if necessary
        if len(seq_tensor.shape) == 2:
            seq_tensor = seq_tensor.unsqueeze(0)

        with torch.no_grad():
            outputs = self.critic_model(seq_tensor)
            prob_grounded = outputs.squeeze(-1).item()
            
        prob_hallucination = 1.0 - prob_grounded
        return round(prob_hallucination, 4)

    def compute_critic_score(
        self,
        query: str,
        retrieved_context: str,
        generated_answer: str,
        seq_tensor: torch.Tensor | None = None,
    ) -> float:
        """
        Evaluate hallucination probability. Uses sequence tensor if provided,
        otherwise falls back to text Jaccard overlap-based groundedness scoring.
        """
        # 1. Tensor scoring path (if tensor is provided and model is loaded)
        if seq_tensor is not None and self.critic_model is not None:
            return self.score_sequence_tensor(seq_tensor)

        # 2. Text-based fallback scoring path
        # Score individual sentences and calculate average grounding
        sentences = self.labeller.label_sentences(generated_answer, retrieved_context)
        if not sentences:
            return 1.0  # Empty answer is treated as unsupported/hallucinated
            
        avg_grounded = sum(is_grounded for _, is_grounded in sentences) / len(sentences)
        prob_hallucination = 1.0 - avg_grounded
        return round(prob_hallucination, 4)

    def score_and_enforce(
        self,
        query: str,
        retrieved_context: str,
        generated_answer: str,
        domain: str,
        seq_tensor: torch.Tensor | None = None,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """
        Execute full guardrail enforcement checking generation risk against thresholds.
        """
        # Resolve threshold
        if threshold is None:
            threshold = self.runtime_threshold_overrides.get(domain)
            
        if threshold is None:
            try:
                domain_cfg = get_domain_config(domain)
                threshold = domain_cfg.critic_hallucination_threshold
            except Exception:
                threshold = 0.5

        # Compute risk score
        critic_score = self.compute_critic_score(
            query=query,
            retrieved_context=retrieved_context,
            generated_answer=generated_answer,
            seq_tensor=seq_tensor,
        )

        fallback_used = critic_score > threshold
        reason = "critic_threshold_crossed" if fallback_used else None

        # Telemetry updates
        with self._lock:
            self.metrics["total_queries"] += 1
            if fallback_used:
                self.metrics["total_fallbacks"] += 1
            n = self.metrics["total_queries"]
            self.metrics["cumulative_critic_score"] += critic_score

        # Structured Audit Logging
        self._write_audit_log(
            domain=domain,
            query=query,
            original_answer=generated_answer,
            critic_score=critic_score,
            threshold=threshold,
            action_taken="fallback" if fallback_used else "continue",
        )

        return {
            "critic_score": critic_score,
            "fallback_used": fallback_used,
            "threshold": round(threshold, 4),
            "reason": reason,
        }

    def get_metrics(self) -> dict[str, Any]:
        """Return a copy of the live guardrail telemetry metrics."""
        with self._lock:
            total = self.metrics["total_queries"]
            avg_score = (
                self.metrics["cumulative_critic_score"] / total if total > 0 else 0.0
            )
            return {
                "total_queries": total,
                "total_fallbacks": self.metrics["total_fallbacks"],
                "fallback_rate": round(self.metrics["total_fallbacks"] / total, 4) if total > 0 else 0.0,
                "average_critic_score": round(avg_score, 4),
            }

    def reset_metrics(self) -> None:
        """Reset live telemetry metrics."""
        with self._lock:
            self.metrics = {
                "total_queries": 0,
                "total_fallbacks": 0,
                "cumulative_critic_score": 0.0,
            }

    def _write_audit_log(
        self,
        domain: str,
        query: str,
        original_answer: str,
        critic_score: float,
        threshold: float,
        action_taken: str,
    ) -> None:
        """Write a single structured JSON line to the audit log."""
        try:
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            log_entry = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                "domain": domain,
                "query": query,
                "original_answer": original_answer,
                "critic_score": critic_score,
                "threshold": threshold,
                "action_taken": action_taken,
            }
            with self.audit_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write guardrail audit log: {e}")
