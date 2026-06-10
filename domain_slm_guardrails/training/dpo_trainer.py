from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Optional
import json
import logging

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from datasets import Dataset
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TrainingArguments = None
    Trainer = None
    DataCollatorForLanguageModeling = None
    Dataset = None

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
except ImportError:  # pragma: no cover
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None
    PeftModel = None

logger = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    """Configuration for DPO alignment training."""

    beta: float = 0.5  # DPO temperature parameter
    learning_rate: float = 5e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 1
    reference_model_path: str = None  # Path to SFT model to use as reference
    policy_model_path: str = None  # Path to SFT model for policy
    base_model: str = "meta-llama/Llama-2-8b"
    adapter_type: str = "lora"
    output_dir: str = "outputs/dpo"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    train_adapter_only: bool = True
    gradient_checkpointing: bool = True
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_seq_length: int = 2048
    save_total_limit: int = 3
    fp16: bool = True
    bf16: bool = False


class DPOTrainer:
    """Orchestrates DPO (Direct Preference Optimization) alignment training."""

    def __init__(self, config: DPOConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.policy_model = None
        self.reference_model = None
        self.tokenizer = None

    def load_models(self) -> None:
        """Load policy and reference models with LoRA adapters."""
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise RuntimeError(
                "transformers is required. Install: pip install transformers peft"
            )

        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.policy_model_path or self.config.base_model
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Load policy model
        logger.info(f"Loading policy model from {self.config.policy_model_path or self.config.base_model}")
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            self.config.policy_model_path or self.config.base_model,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load or setup LoRA for policy model
        if self.config.policy_model_path and (Path(self.config.policy_model_path) / "adapter_config.json").exists():
            # Load existing LoRA adapter
            if PeftModel is None:
                raise RuntimeError("peft is required")
            self.policy_model = PeftModel.from_pretrained(
                self.policy_model,
                self.config.policy_model_path,
            )
            logger.info(f"Loaded LoRA adapter from {self.config.policy_model_path}")
        else:
            # Create new LoRA adapter
            if get_peft_model is None:
                raise RuntimeError("peft is required")

            peft_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj"],
            )
            self.policy_model = get_peft_model(self.policy_model, peft_config)

        # Load reference model (frozen)
        logger.info(f"Loading reference model from {self.config.reference_model_path or self.config.base_model}")
        self.reference_model = AutoModelForCausalLM.from_pretrained(
            self.config.reference_model_path or self.config.base_model,
            device_map="auto",
            trust_remote_code=True,
        )
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad = False

        logger.info("Models loaded successfully")

    def prepare_dataset(
        self,
        examples: Iterable[dict[str, str]],
    ) -> Dataset:
        """Prepare dataset for DPO training."""
        if Dataset is None:
            raise RuntimeError("datasets is required. Install: pip install datasets")

        # Convert DPO pairs to training format
        formatted_examples = []
        for example in examples:
            query = example.get("query", "")
            chosen = example.get("chosen", "")
            rejected = example.get("rejected", "")

            formatted_examples.append({
                "prompt": f"Query: {query}\n\nAnswer:",
                "chosen": chosen,
                "rejected": rejected,
            })

        dataset = Dataset.from_dict({
            "prompt": [ex["prompt"] for ex in formatted_examples],
            "chosen": [ex["chosen"] for ex in formatted_examples],
            "rejected": [ex["rejected"] for ex in formatted_examples],
        })

        # Tokenize all three parts
        def tokenize_function(examples):
            prompts = self.tokenizer(
                examples["prompt"],
                max_length=self.config.max_seq_length,
                truncation=True,
                padding="max_length",
            )
            chosen = self.tokenizer(
                examples["chosen"],
                max_length=self.config.max_seq_length,
                truncation=True,
                padding="max_length",
            )
            rejected = self.tokenizer(
                examples["rejected"],
                max_length=self.config.max_seq_length,
                truncation=True,
                padding="max_length",
            )

            return {
                "prompt_input_ids": prompts["input_ids"],
                "prompt_attention_mask": prompts["attention_mask"],
                "chosen_input_ids": chosen["input_ids"],
                "chosen_attention_mask": chosen["attention_mask"],
                "rejected_input_ids": rejected["input_ids"],
                "rejected_attention_mask": rejected["attention_mask"],
            }

        tokenized = dataset.map(tokenize_function, batched=True)
        logger.info(f"Dataset prepared: {len(tokenized)} pairs")

        return tokenized

    def _compute_dpo_loss(
        self,
        policy_chosen_logits: torch.Tensor,
        policy_rejected_logits: torch.Tensor,
        reference_chosen_logits: torch.Tensor,
        reference_rejected_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DPO loss.
        
        DPO loss = -log(sigmoid(beta * (log_pi(y_c) - log_pi(y_r) - log_ref(y_c) + log_ref(y_r))))
        """
        # Log probabilities (average per token)
        policy_chosen_lp = -F.cross_entropy(
            policy_chosen_logits.view(-1, policy_chosen_logits.size(-1)),
            torch.arange(policy_chosen_logits.size(1)).expand(policy_chosen_logits.size(0), -1).to(self.device),
            reduction="none",
        ).view(policy_chosen_logits.size(0), -1).sum(dim=1)
        
        policy_rejected_lp = -F.cross_entropy(
            policy_rejected_logits.view(-1, policy_rejected_logits.size(-1)),
            torch.arange(policy_rejected_logits.size(1)).expand(policy_rejected_logits.size(0), -1).to(self.device),
            reduction="none",
        ).view(policy_rejected_logits.size(0), -1).sum(dim=1)

        reference_chosen_lp = -F.cross_entropy(
            reference_chosen_logits.view(-1, reference_chosen_logits.size(-1)),
            torch.arange(reference_chosen_logits.size(1)).expand(reference_chosen_logits.size(0), -1).to(self.device),
            reduction="none",
        ).view(reference_chosen_logits.size(0), -1).sum(dim=1)

        reference_rejected_lp = -F.cross_entropy(
            reference_rejected_logits.view(-1, reference_rejected_logits.size(-1)),
            torch.arange(reference_rejected_logits.size(1)).expand(reference_rejected_logits.size(0), -1).to(self.device),
            reduction="none",
        ).view(reference_rejected_logits.size(0), -1).sum(dim=1)

        # DPO loss
        log_ratio = (policy_chosen_lp - policy_rejected_lp) - (reference_chosen_lp - reference_rejected_lp)
        loss = -F.logsigmoid(self.config.beta * log_ratio).mean()

        return loss

    def train(
        self,
        train_examples: Iterable[dict[str, str]],
        validation_examples: Optional[Iterable[dict[str, str]]] = None,
    ) -> dict[str, object]:
        """Execute DPO training."""
        if self.policy_model is None:
            self.load_models()

        train_dataset = self.prepare_dataset(train_examples)
        val_dataset = self.prepare_dataset(validation_examples) if validation_examples else None

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_train_epochs,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            eval_strategy="steps" if val_dataset else "no",
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            gradient_checkpointing=self.config.gradient_checkpointing,
            save_total_limit=self.config.save_total_limit,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            optim="paged_adamw_8bit" if self.config.fp16 or self.config.bf16 else "adamw_torch",
        )

        # Create trainer
        trainer = Trainer(
            model=self.policy_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
        )

        logger.info("Starting DPO training...")
        train_result = trainer.train()

        # Save adapter
        self.save_adapter(self.config.output_dir)
        self.save_config(Path(self.config.output_dir) / "dpo_config.json")

        logger.info(f"DPO training completed. Model saved to {self.config.output_dir}")

        return {
            "status": "completed",
            "output_dir": self.config.output_dir,
            "metrics": train_result.metrics,
        }

    def save_adapter(self, path: Path | str) -> Path:
        """Save LoRA adapter weights."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.policy_model is not None:
            self.policy_model.save_pretrained(path)
            logger.info(f"DPO adapter saved to {path}")

        return path

    def save_config(self, path: Path | str) -> Path:
        """Save DPO configuration."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2)

        logger.info(f"Config saved to {path}")
        return path

    def export_merged_model(self, output_path: Path | str) -> Path:
        """Export merged model (base + LoRA weights merged)."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.policy_model is None:
            raise RuntimeError("No model loaded")

        logger.info("Merging DPO adapter with base model...")
        merged_model = self.policy_model.merge_and_unload()
        merged_model.save_pretrained(output_path)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_path)

        logger.info(f"Merged model saved to {output_path}")
        return output_path
