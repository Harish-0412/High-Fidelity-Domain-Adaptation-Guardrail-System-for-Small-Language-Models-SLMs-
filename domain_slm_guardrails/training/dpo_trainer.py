from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional
import json
import logging

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TrainingArguments = None
    Trainer = None

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
except ImportError:  # pragma: no cover
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_int8_training = None


@dataclass
class DPOConfig:
    beta: float = 0.5
    learning_rate: float = 1e-5
    batch_size: int = 8
    num_train_epochs: int = 1
    reference_model: str = "gpt2"
    adapter_type: str = "lora"
    output_dir: str = "outputs/dpo"
    train_adapter_only: bool = True
    gradient_checkpointing: bool = False
    save_steps: int = 1000
    logging_steps: int = 100


class DPOTrainer:
    """Orchestrates DPO alignment training and adapter checkpointing."""

    def __init__(self, policy_model_name: str, config: DPOConfig, device: str = "cpu"):
        self.policy_model_name = policy_model_name
        self.config = config
        self.device = device
        self.policy_model = None
        self.reference_model = None
        self.tokenizer = None

    def load_models(self) -> None:
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise RuntimeError(
                "transformers is required to load DPO models. Install transformers and peft."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.policy_model_name)
        self.policy_model = AutoModelForCausalLM.from_pretrained(self.policy_model_name)
        self.reference_model = AutoModelForCausalLM.from_pretrained(self.config.reference_model)

        if self.config.adapter_type == "lora" and get_peft_model is not None:
            if prepare_model_for_int8_training is not None:
                self.policy_model = prepare_model_for_int8_training(self.policy_model)
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
            )
            self.policy_model = get_peft_model(self.policy_model, peft_config)

        logging.info("Loaded policy model %s and reference model %s", self.policy_model_name, self.config.reference_model)

    def train(
        self,
        train_dataset: Iterable[dict[str, str]],
        validation_dataset: Optional[Iterable[dict[str, str]]] = None,
    ) -> dict[str, object]:
        if self.policy_model is None or self.reference_model is None:
            self.load_models()

        if Trainer is None or TrainingArguments is None:
            raise RuntimeError(
                "transformers Trainer is required for the DPO training loop."
            )

        train_args = TrainingArguments(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_train_epochs,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            fp16=self.device != "cpu",
            evaluation_strategy="steps" if validation_dataset else "no",
            save_total_limit=2,
        )

        # The actual DPO loss implementation should be integrated here.
        # This template can be adapted to TRL DPOTrainer, custom loss, or PEFT.
        trainer = Trainer(
            model=self.policy_model,
            args=train_args,
            train_dataset=list(train_dataset),
            eval_dataset=list(validation_dataset) if validation_dataset is not None else None,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        self.save_checkpoint(self.config.output_dir)

        metrics = trainer.state.log_history[-1] if trainer.state.log_history else {}
        return {
            "status": "completed",
            "output_dir": self.config.output_dir,
            "metrics": metrics,
        }

    def evaluate(self, eval_dataset: Iterable[dict[str, str]]) -> dict[str, object]:
        if self.policy_model is None:
            self.load_models()

        return {
            "cases": len(list(eval_dataset)),
            "groundedness_score": None,
            "notes": "Implement evaluation using groundedness metrics from the comparator module.",
        }

    def save_checkpoint(self, path: Path | str) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.policy_model is not None:
            self.policy_model.save_pretrained(path)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(path)
        return path

    def export_config(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.config.__dict__, handle, indent=2)
        return path
