"""QLoRA SFT Trainer: Supervised fine-tuning using QLoRA parameter-efficient adaptation."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable, Optional
import json
import logging

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from datasets import Dataset
except ImportError:  # pragma: no cover
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TrainingArguments = None
    Trainer = None
    DataCollatorForLanguageModeling = None
    Dataset = None

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from bitsandbytes.optim import AdamW8bit
except ImportError:  # pragma: no cover
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None
    AdamW8bit = None

logger = logging.getLogger(__name__)


@dataclass
class QLoRASFTConfig:
    """Configuration for QLoRA SFT training."""

    model_name: str = "meta-llama/Llama-2-8b"
    output_dir: str = "outputs/sft"
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3
    max_seq_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_checkpointing: bool = True
    save_total_limit: int = 3
    bf16: bool = False


class QLoRASFTTrainer:
    """Train a model using QLoRA for parameter-efficient supervised fine-tuning."""

    def __init__(self, config: QLoRASFTConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.model = None
        self.tokenizer = None
        self.peft_model = None

    def load_model_and_tokenizer(self) -> None:
        """Load base model and tokenizer with 4-bit quantization."""
        if AutoModelForCausalLM is None:
            raise RuntimeError(
                "transformers is required. Install: pip install transformers peft bitsandbytes"
            )

        logger.info(f"Loading model: {self.config.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Load model with 4-bit quantization if enabled
        if self.config.load_in_4bit:
            try:
                import bitsandbytes as bnb

                bnb_config = {
                    "load_in_4bit": True,
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.float16
                    if self.config.bnb_4bit_compute_dtype == "float16"
                    else torch.bfloat16,
                }

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            except Exception as e:
                logger.warning(f"Failed to load with quantization: {e}. Loading without 4bit.")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    device_map="auto",
                    trust_remote_code=True,
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map="auto",
                trust_remote_code=True,
            )

        # Prepare for kbit training
        if prepare_model_for_kbit_training is not None:
            self.model = prepare_model_for_kbit_training(self.model)

        # Setup LoRA
        if get_peft_model is None:
            raise RuntimeError("peft is required. Install: pip install peft")

        peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.config.lora_target_modules,
        )

        self.peft_model = get_peft_model(self.model, peft_config)
        self.peft_model.print_trainable_parameters()

        logger.info("Model and LoRA config loaded successfully")

    def prepare_dataset(
        self,
        examples: Iterable[dict[str, str]],
        split_ratio: float = 0.9,
    ) -> tuple[Dataset, Dataset]:
        """
        Prepare HuggingFace Dataset from SFT examples.
        
        Args:
            examples: Iterable of dicts with 'query' and 'answer' fields.
            split_ratio: Train/val split ratio.
        
        Returns:
            Tuple of (train_dataset, val_dataset).
        """
        if Dataset is None:
            raise RuntimeError("datasets is required. Install: pip install datasets")

        # Convert examples to format expected by Trainer
        formatted_examples = []
        for example in examples:
            query = example.get("query", "")
            answer = example.get("answer", "")
            
            # Combine query and answer for training
            text = f"Query: {query}\n\nAnswer: {answer}"
            formatted_examples.append({"text": text})

        # Create HF Dataset
        dataset = Dataset.from_dict({"text": [ex["text"] for ex in formatted_examples]})

        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                max_length=self.config.max_seq_length,
                truncation=True,
                padding="max_length",
            )

        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
        )

        # Split
        split_data = tokenized.train_test_split(train_size=split_ratio)
        train_dataset = split_data["train"]
        val_dataset = split_data["test"]

        logger.info(
            f"Dataset prepared: {len(train_dataset)} train, {len(val_dataset)} val"
        )

        return train_dataset, val_dataset

    def train(
        self,
        train_examples: Iterable[dict[str, str]],
        val_examples: Optional[Iterable[dict[str, str]]] = None,
    ) -> dict[str, object]:
        """
        Execute QLoRA SFT training.
        
        Args:
            train_examples: Training examples with 'query' and 'answer'.
            val_examples: Optional validation examples.
        
        Returns:
            Training results dict with metrics and output path.
        """
        if self.peft_model is None:
            self.load_model_and_tokenizer()

        # Prepare datasets
        train_dataset, val_dataset_auto = self.prepare_dataset(train_examples)

        if val_examples:
            val_dataset = self.prepare_dataset(val_examples, split_ratio=1.0)[0]
        else:
            val_dataset = val_dataset_auto

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_train_epochs,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            gradient_checkpointing=self.config.gradient_checkpointing,
            save_total_limit=self.config.save_total_limit,
            bf16=self.config.bf16,
            fp16=not self.config.bf16,
            optim="paged_adamw_8bit",
        )

        # Create trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorForLanguageModeling(
                self.tokenizer, mlm=False
            ),
            tokenizer=self.tokenizer,
        )

        # Train
        logger.info("Starting QLoRA SFT training...")
        train_result = trainer.train()

        # Save final model
        self.save_adapter(self.config.output_dir)
        self.save_config(Path(self.config.output_dir) / "config.json")

        logger.info(f"Training completed. Model saved to {self.config.output_dir}")

        return {
            "status": "completed",
            "output_dir": self.config.output_dir,
            "metrics": train_result.metrics,
            "model_name": self.config.model_name,
        }

    def save_adapter(self, path: Path | str) -> Path:
        """Save LoRA adapter weights."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.peft_model is not None:
            self.peft_model.save_pretrained(path)
            logger.info(f"LoRA adapter saved to {path}")

        return path

    def save_config(self, path: Path | str) -> Path:
        """Save training configuration."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2)

        logger.info(f"Config saved to {path}")
        return path

    def load_adapter(self, adapter_path: Path | str) -> None:
        """Load a saved LoRA adapter."""
        adapter_path = Path(adapter_path)

        if self.model is None:
            self.load_model_and_tokenizer()

        if get_peft_model is None:
            raise RuntimeError("peft is required")

        # Load adapter config from saved model
        from peft import PeftModel

        self.peft_model = PeftModel.from_pretrained(self.model, adapter_path)
        logger.info(f"LoRA adapter loaded from {adapter_path}")

    def export_merged_model(self, output_path: Path | str) -> Path:
        """
        Export merged model (base model + LoRA weights merged).
        
        This creates a single, standalone model that doesn't require
        loading the adapter separately.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.peft_model is None:
            raise RuntimeError("No model loaded. Call load_adapter first.")

        logger.info("Merging LoRA weights with base model...")
        
        merged_model = self.peft_model.merge_and_unload()
        merged_model.save_pretrained(output_path)
        
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_path)

        logger.info(f"Merged model saved to {output_path}")
        return output_path
