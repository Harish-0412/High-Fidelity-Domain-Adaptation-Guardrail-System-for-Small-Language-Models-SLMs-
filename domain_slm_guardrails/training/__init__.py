"""Training package for QLoRA SFT and DPO workflows.

Modules:
- sft_dataset.py: creates supervised fine-tuning datasets from domain chunks.
- sft_trainer.py: orchestrates QLoRA supervised fine-tuning.
- dpo_generator.py: generates preference pairs for aligned ranking.
- dpo_trainer.py: orchestrates DPO alignment training.
- pipeline.py: end-to-end training orchestration (SFT -> DPO).
"""

