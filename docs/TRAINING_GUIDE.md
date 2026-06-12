# Training and Evaluation Guide

This guide covers the Week 3-4 implementation of QLoRA SFT training, DPO alignment, adapter export, and production groundedness comparison.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. SFT Dataset Creation                                         │
│     ├─ Load domain chunks                                        │
│     ├─ Generate queries from chunks (templated)                 │
│     └─ Mix with general data (20% default)                      │
│                                                                   │
│  2. QLoRA SFT Training                                           │
│     ├─ Load base model with 4-bit quantization                  │
│     ├─ Apply LoRA adapters (r=16, α=32)                         │
│     ├─ Train on mixed dataset                                   │
│     └─ Save adapter weights                                     │
│                                                                   │
│  3. DPO Preference Pair Generation                              │
│     ├─ Load SFT examples                                        │
│     ├─ Generate rejected variants (4 strategies)                │
│     └─ Export as JSONL preference pairs                         │
│                                                                   │
│  4. DPO Alignment Training                                      │
│     ├─ Load SFT model as policy                                 │
│     ├─ Load base model as reference                             │
│     ├─ Compute DPO loss (β-weighted preference ranking)         │
│     └─ Save aligned adapter                                     │
│                                                                   │
│  5. Evaluation & Comparison                                     │
│     ├─ Baseline model evaluation                                │
│     ├─ SFT model evaluation                                     │
│     ├─ DPO model evaluation                                     │
│     └─ Groundedness comparison (SFT vs DPO)                     │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. SFT Dataset Builder (`sft_dataset.py`)

Creates supervised fine-tuning datasets from domain chunks.

**Key Classes:**
- `SFTExample`: Individual training example with query, answer, and citations
- `SFTDatasetBuilder`: Creates examples from chunks and handles data mixing
- `GeneralDataLoader`: Loads general-purpose data for mixing

**Features:**
- Automatic query generation from chunks using templates
- Domain + general data mixing (configurable ratio)
- Train/val splitting
- JSONL export/import

**Example:**
```python
from services.training.sft_dataset import SFTDatasetBuilder

builder = SFTDatasetBuilder()

# Create from domain chunks
chunks = read_chunks_jsonl("data/chunks.jsonl")
examples = builder.create_from_chunks(chunks)

# Mix with general data
general_data = GeneralDataLoader.create_dummy_general_data(size=100)
mixed = builder.mix_with_general_data(examples, general_data, general_ratio=0.2)

# Export
builder.export_jsonl(mixed, "data/sft_dataset.jsonl")
```

### 2. QLoRA SFT Trainer (`sft_trainer.py`)

Implements efficient supervised fine-tuning using QLoRA.

**Key Classes:**
- `QLoRASFTConfig`: Configuration for training
- `QLoRASFTTrainer`: Training orchestration

**Features:**
- 4-bit quantization support
- LoRA adapters (r=16, α=32 by default)
- Gradient checkpointing
- Mixed precision training (fp16/bf16)
- Adapter and merged model export

**Configuration:**
```python
from services.training.sft_trainer import QLoRASFTTrainer, QLoRASFTConfig

config = QLoRASFTConfig(
    model_name="meta-llama/Llama-2-8b",
    output_dir="outputs/sft_model",
    learning_rate=2e-4,
    batch_size=4,
    num_train_epochs=3,
    lora_r=16,
    lora_alpha=32,
)

trainer = QLoRASFTTrainer(config)
trainer.load_model_and_tokenizer()

# Train on examples
train_dataset, val_dataset = trainer.prepare_dataset(examples)
results = trainer.train(examples)

# Export
trainer.save_adapter("outputs/sft_adapter")
trainer.export_merged_model("outputs/sft_merged")
```

### 3. DPO Preference Generator (`dpo_generator.py`)

Creates preference pairs for DPO alignment training.

**Key Classes:**
- `DPOPreferencePair`: Preference pair with rejection strategy
- `DPOPreferenceGenerator`: Generates preference pairs

**Rejection Strategies:**
- `weakly_cited`: Weakly supported answers (partial citations)
- `hallucinated`: Fabricated details added to answers
- `incomplete`: Truncated or missing key information
- `overly_verbose`: Unnecessarily long answers with redundancy

**Example:**
```python
from services.training.dpo_generator import DPOPreferenceGenerator

generator = DPOPreferenceGenerator()

# Generate from SFT examples
sft_examples = builder.import_jsonl("data/sft_dataset.jsonl")
pairs = generator.generate_from_sft_examples(
    [ex.to_dict() for ex in sft_examples],
    max_rejections_per_example=1,
)

# Export
generator.export_jsonl(pairs, "data/dpo_pairs.jsonl")
```

### 4. DPO Trainer (`dpo_trainer.py`)

Implements Direct Preference Optimization for alignment.

**Key Classes:**
- `DPOConfig`: Configuration for DPO training
- `DPOTrainer`: DPO training orchestration

**Features:**
- Policy model with LoRA adapters
- Frozen reference model
- DPO loss computation with β temperature parameter
- Adapter and merged model export

**DPO Loss:**
```
L_DPO = -log(σ(β * (log π(y_c) - log π(y_r) - log π_ref(y_c) + log π_ref(y_r))))
```

Where:
- π = policy model
- π_ref = reference model
- y_c = chosen response
- y_r = rejected response
- β = temperature parameter (controls preference strength)

**Example:**
```python
from services.training.dpo_trainer import DPOTrainer, DPOConfig

config = DPOConfig(
    beta=0.5,
    learning_rate=5e-5,
    batch_size=8,
    num_train_epochs=1,
    policy_model_path="outputs/sft_model",
    reference_model_path="outputs/sft_model",
    base_model="meta-llama/Llama-2-8b",
    output_dir="outputs/dpo_model",
)

trainer = DPOTrainer(config)
trainer.load_models()

# Load DPO pairs
with open("data/dpo_pairs.jsonl") as f:
    dpo_pairs = [json.loads(line) for line in f]

results = trainer.train(dpo_pairs)

# Export
trainer.export_merged_model("outputs/dpo_merged")
```

### 5. Training Pipeline (`pipeline.py`)

End-to-end orchestration combining SFT and DPO.

**Example:**
```python
from services.training.pipeline import run_training_pipeline

results = run_training_pipeline(
    domain_id="medical_prescription",
    base_model="meta-llama/Llama-2-8b",
    output_dir="outputs/training",
    train_sft=True,
    train_dpo=True,
)
```

### 6. Baseline Model Evaluation (`baseline_eval.py`)

Evaluates model performance on benchmark cases.

**Key Classes:**
- `ModelEvalCase`: Evaluation case with expected outputs
- `ModelEvalResult`: Result of model evaluation
- `BaselineModelEvaluator`: Single model evaluation
- `MultiModelComparison`: Compare multiple models

**Metrics:**
- Term match rate
- Factual consistency
- Hallucination score (estimated)
- Conciseness score
- Citation coverage
- Latency

**Example:**
```python
from services.evaluation.baseline_eval import BaselineModelEvaluator

# Evaluate single model
evaluator = BaselineModelEvaluator("meta-llama/Llama-2-8b")
results = evaluator.evaluate_cases(eval_cases)
summary = evaluator.summarize_results(results)

# Compare multiple models
from services.evaluation.baseline_eval import MultiModelComparison

models = {
    "base": "meta-llama/Llama-2-8b",
    "sft": "outputs/sft_merged",
    "dpo": "outputs/dpo_merged",
}
comparison = MultiModelComparison(models)
all_results = comparison.evaluate_all(eval_cases)
summary = comparison.compare_summary(all_results)
```

### 7. Groundedness Comparison (`groundedness_comparator.py`)

Compares SFT baseline vs DPO policy on groundedness metrics.

**Key Classes:**
- `GroundednessCase`: Case with baseline and policy outputs
- `GroundednessMetrics`: Computed metrics
- `GroundednessComparator`: Comparison orchestration

**Metrics per output:**
- Citation density
- Grounding score (RAG-based)
- Factual consistency
- Hallucination penalty
- Conciseness
- Fallback usage

**Example:**
```python
from services.evaluation.groundedness_comparator import (
    GroundednessCase,
    GroundednessComparator,
)

cases = [
    GroundednessCase(
        id="case_1",
        query="Question?",
        baseline_answer="Baseline response",
        policy_answer="Policy response with more citations",
        baseline_citations=[],
        policy_citations=[{"source_id": "doc1"}],
        baseline_guardrail={"rag_grounded": False},
        policy_guardrail={"rag_grounded": True, "critic_score": 0.15},
    ),
]

comparator = GroundednessComparator(cases)
results = comparator.compare()

# Export
comparator.export_json("results/comparison.json")
comparator.export_csv("results/comparison.csv")
comparator.export_markdown("results/comparison.md")
```

## Running Training and Evaluation

### Install Dependencies

```bash
pip install -e ".[training,evaluation]"
```

### Run Training Pipeline

```bash
python scripts/run_training_pipeline.py \
    --domain medical_prescription \
    --base-model meta-llama/Llama-2-8b \
    --output-dir outputs/training
```

### Run Baseline Evaluation

```bash
# Single model
python scripts/run_baseline_eval.py \
    --model outputs/sft_merged \
    --output-dir outputs/baseline_eval

# Compare multiple models
python scripts/run_baseline_eval.py \
    --models meta-llama/Llama-2-8b outputs/sft_merged outputs/dpo_merged \
    --output-dir outputs/comparison_eval
```

### Run Groundedness Comparison

```bash
python scripts/run_groundedness_comparison.py \
    --output-dir outputs/groundedness \
    --format markdown
```

## Testing

Run the comprehensive test suite:

```bash
# SFT dataset tests
pytest tests/test_sft_training.py -v

# DPO training tests
pytest tests/test_dpo_training.py -v

# Groundedness comparison tests
pytest tests/test_groundedness_comparison.py -v

# All tests
pytest tests/ -v
```

## Output Structure

```
outputs/
├── training/
│   ├── sft_dataset.jsonl          # Generated SFT examples
│   ├── sft_model/                 # QLoRA adapter weights
│   ├── sft_merged/                # Merged model (base + LoRA)
│   ├── dpo_dataset.jsonl          # Generated preference pairs
│   ├── dpo_model/                 # DPO adapter weights
│   └── dpo_merged/                # Merged model (final)
├── baseline_eval/
│   └── results.json               # Baseline evaluation results
├── comparison_eval/
│   ├── model1_results.json
│   ├── model2_results.json
│   └── comparison_summary.json
└── groundedness/
    ├── comparison.json
    ├── comparison.csv
    └── comparison.md
```

## Configuration Tuning

### QLoRA SFT Training

**For production (medical prescription):**
```python
QLoRASFTConfig(
    learning_rate=2e-4,
    batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    lora_r=16,
    lora_alpha=32,
    max_seq_length=2048,
)
```

**For faster iteration (testing):**
```python
QLoRASFTConfig(
    learning_rate=5e-4,
    batch_size=8,
    num_train_epochs=1,
    lora_r=8,
    max_seq_length=1024,
)
```

### DPO Training

**For strong alignment:**
```python
DPOConfig(
    beta=0.5,  # Higher = stronger preference enforcement
    learning_rate=5e-5,
    num_train_epochs=2,
)
```

**For data-efficient alignment:**
```python
DPOConfig(
    beta=0.1,  # Lower = softer training
    learning_rate=1e-4,
    batch_size=16,
    num_train_epochs=1,
)
```

## Performance Targets (from PLAN.md)

- QLoRA training latency: ~2-4 hours on single GPU
- DPO training latency: ~1-2 hours on single GPU
- Inference latency: <100ms per query (with guardrails)
- SFT baseline test pass rate: >80%
- DPO policy improvement: >10% vs SFT baseline
- Groundedness score improvement: >15% on policy vs baseline

## Troubleshooting

### GPU Memory Issues

If you get OOM errors:
1. Reduce `batch_size` (try 2 or 4)
2. Reduce `max_seq_length` (try 1024)
3. Enable `gradient_checkpointing` (already default)
4. Use smaller LoRA `lora_r` (try 8)

### Poor Training Convergence

1. Check learning rate (default 2e-4 for SFT, 5e-5 for DPO)
2. Verify data quality (check example.json files)
3. Increase epochs if data is small
4. Adjust warmup_steps for longer ramp-up

### Hallucination Issues

1. Increase DPO `beta` (0.5 → 1.0)
2. Use more general data in mixing (0.2 → 0.3)
3. Add more rejection examples (1 → 2 per SFT example)

## Next Steps (Week 5-6)

- Hidden-state collection from middle layers
- Token-level hallucination labeling
- Critic model training for early hallucination detection
- Integration with guardrail system for constrained decoding
