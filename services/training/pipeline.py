"""Training Orchestration: End-to-end SFT and DPO pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import json
import logging

from services.core.domain_registry import get_domain_config
from ingestion.pipeline import read_chunks_jsonl
from services.training.sft_dataset import (
    SFTDatasetBuilder,
    GeneralDataLoader,
)
from services.training.sft_trainer import QLoRASFTTrainer, QLoRASFTConfig
from services.training.dpo_generator import DPOPreferenceGenerator
from services.training.dpo_trainer import DPOTrainer, DPOConfig

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Orchestrates end-to-end SFT and DPO training."""

    def __init__(
        self,
        domain_id: str,
        base_model: str = "meta-llama/Llama-2-8b",
        output_dir: str = "outputs/training",
        seed: int = 42,
    ):
        self.domain_id = domain_id
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.seed = seed

    def run_full_pipeline(
        self,
        train_sft: bool = True,
        train_dpo: bool = True,
        general_data_ratio: float = 0.2,
    ) -> dict[str, object]:
        """
        Run complete training pipeline: SFT -> DPO.
        
        Args:
            train_sft: Whether to train SFT model.
            train_dpo: Whether to train DPO model.
            general_data_ratio: Ratio of general data to mix with domain data.
        
        Returns:
            Pipeline results dict.
        """
        results = {
            "domain": self.domain_id,
            "base_model": self.base_model,
            "pipeline_steps": [],
        }

        try:
            # Load domain config
            logger.info(f"Loading domain config: {self.domain_id}")
            domain_config = get_domain_config(self.domain_id)

            # Step 1: Create SFT dataset
            logger.info("Step 1: Creating SFT dataset...")
            sft_dataset_path = self._create_sft_dataset(
                domain_config,
                general_data_ratio,
            )
            results["pipeline_steps"].append({
                "step": "sft_dataset_creation",
                "status": "completed",
                "output": str(sft_dataset_path),
            })

            # Step 2: Train SFT model
            if train_sft:
                logger.info("Step 2: Training SFT model...")
                sft_output = self._train_sft(sft_dataset_path)
                results["pipeline_steps"].append({
                    "step": "sft_training",
                    "status": "completed",
                    "output": sft_output,
                })

                sft_model_path = sft_output["output_dir"]
            else:
                logger.info("Step 2: Skipping SFT training (use_sft=False)")
                sft_model_path = self.base_model

            # Step 3: Create DPO dataset
            logger.info("Step 3: Creating DPO preference pairs...")
            dpo_dataset_path = self._create_dpo_dataset(
                domain_config,
                sft_dataset_path,
            )
            results["pipeline_steps"].append({
                "step": "dpo_dataset_creation",
                "status": "completed",
                "output": str(dpo_dataset_path),
            })

            # Step 4: Train DPO model
            if train_dpo:
                logger.info("Step 4: Training DPO model...")
                dpo_output = self._train_dpo(
                    sft_model_path,
                    dpo_dataset_path,
                )
                results["pipeline_steps"].append({
                    "step": "dpo_training",
                    "status": "completed",
                    "output": dpo_output,
                })

                final_model_path = dpo_output["output_dir"]
            else:
                logger.info("Step 4: Skipping DPO training (use_dpo=False)")
                final_model_path = sft_model_path

            results["final_model_path"] = final_model_path
            results["status"] = "completed"

            logger.info(f"Training pipeline completed. Final model: {final_model_path}")

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}", exc_info=True)
            results["status"] = "failed"
            results["error"] = str(e)

        return results

    def _create_sft_dataset(
        self,
        domain_config,
        general_data_ratio: float,
    ) -> Path:
        """Create SFT dataset from domain chunks and general data."""
        # Load domain chunks
        logger.info(f"Loading chunks from {domain_config.chunks_path}")
        chunks = read_chunks_jsonl(domain_config.chunks_path)

        # Generate SFT examples from chunks
        builder = SFTDatasetBuilder(seed=self.seed)
        templates = domain_config.settings.get("prompt_templates")
        domain_examples = builder.create_from_chunks(chunks, templates=templates)
        logger.info(f"Generated {len(domain_examples)} domain examples")

        # Load or create general data
        if general_data_ratio > 0:
            logger.info(f"Loading general data (ratio={general_data_ratio})")
            general_data = GeneralDataLoader.create_dummy_general_data(
                size=max(100, int(len(domain_examples) * general_data_ratio / (1.0 - general_data_ratio)))
            )
            
            mixed_examples = builder.mix_with_general_data(
                domain_examples,
                general_data,
                general_ratio=general_data_ratio,
            )
            logger.info(f"Mixed dataset: {len(mixed_examples)} total examples")
        else:
            mixed_examples = domain_examples

        # Export dataset
        output_path = self.output_dir / "sft_dataset.jsonl"
        builder.export_jsonl(mixed_examples, output_path)

        return output_path

    def _train_sft(self, dataset_path: Path | str) -> dict[str, object]:
        """Train QLoRA SFT model."""
        dataset_path = Path(dataset_path)

        # Load examples
        builder = SFTDatasetBuilder()
        examples = builder.import_jsonl(dataset_path)

        # Split train/val
        train_examples, val_examples = builder.split_train_val(examples, train_ratio=0.9)

        # Configure SFT trainer
        config = QLoRASFTConfig(
            model_name=self.base_model,
            output_dir=str(self.output_dir / "sft_model"),
            learning_rate=2e-4,
            batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            max_seq_length=2048,
        )

        # Train
        trainer = QLoRASFTTrainer(config)
        result = trainer.train(
            [ex.to_dict() for ex in train_examples],
            [ex.to_dict() for ex in val_examples],
        )

        return result

    def _create_dpo_dataset(
        self,
        domain_config,
        sft_dataset_path: Path | str,
    ) -> Path:
        """Create DPO preference pairs from SFT dataset."""
        # Load SFT examples
        builder = SFTDatasetBuilder()
        sft_examples = builder.import_jsonl(sft_dataset_path)

        # Generate DPO pairs
        dpo_generator = DPOPreferenceGenerator(seed=self.seed)
        dpo_pairs = dpo_generator.generate_from_sft_examples(
            [ex.to_dict() for ex in sft_examples],
            max_rejections_per_example=1,
        )

        # Export
        output_path = self.output_dir / "dpo_dataset.jsonl"
        dpo_generator.export_jsonl(dpo_pairs, output_path)

        logger.info(f"Generated {len(dpo_pairs)} DPO preference pairs")

        return output_path

    def _train_dpo(
        self,
        sft_model_path: str,
        dpo_dataset_path: Path | str,
    ) -> dict[str, object]:
        """Train DPO model."""
        dpo_dataset_path = Path(dpo_dataset_path)

        # Load DPO examples
        dpo_pairs = []
        with dpo_dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    dpo_pairs.append(json.loads(line))

        # Configure DPO trainer
        config = DPOConfig(
            beta=0.5,
            learning_rate=5e-5,
            batch_size=8,
            num_train_epochs=1,
            policy_model_path=sft_model_path,
            reference_model_path=sft_model_path,
            base_model=self.base_model,
            output_dir=str(self.output_dir / "dpo_model"),
        )

        # Train
        trainer = DPOTrainer(config)
        result = trainer.train(dpo_pairs)

        return result


def run_training_pipeline(
    domain_id: str,
    base_model: str = "meta-llama/Llama-2-8b",
    output_dir: str = "outputs/training",
    train_sft: bool = True,
    train_dpo: bool = True,
) -> dict[str, object]:
    """Convenience function to run full training pipeline."""
    pipeline = TrainingPipeline(
        domain_id=domain_id,
        base_model=base_model,
        output_dir=output_dir,
    )

    return pipeline.run_full_pipeline(
        train_sft=train_sft,
        train_dpo=train_dpo,
    )
