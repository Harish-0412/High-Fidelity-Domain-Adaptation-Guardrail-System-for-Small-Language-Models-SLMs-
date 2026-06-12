#!/usr/bin/env python3
"""Quick validation script to test all new module imports."""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all new modules can be imported."""
    try:
        print("Testing imports...")
        
        print("  ✓ Importing SFT dataset builder...", end="")
        from services.training.sft_dataset import SFTDatasetBuilder, GeneralDataLoader
        print(" OK")
        
        print("  ✓ Importing SFT trainer...", end="")
        from services.training.sft_trainer import QLoRASFTTrainer, QLoRASFTConfig
        print(" OK")
        
        print("  ✓ Importing DPO generator...", end="")
        from services.training.dpo_generator import DPOPreferenceGenerator, DPOPreferencePair
        print(" OK")
        
        print("  ✓ Importing DPO trainer...", end="")
        from services.training.dpo_trainer import DPOTrainer, DPOConfig
        print(" OK")
        
        print("  ✓ Importing training pipeline...", end="")
        from services.training.pipeline import TrainingPipeline, run_training_pipeline
        print(" OK")
        
        print("  ✓ Importing baseline evaluator...", end="")
        from services.evaluation.baseline_eval import (
            BaselineModelEvaluator,
            MultiModelComparison,
            ModelEvalCase,
        )
        print(" OK")
        
        print("  ✓ Importing groundedness comparator...", end="")
        from services.evaluation.groundedness_comparator import (
            GroundednessComparator,
            GroundednessCase,
        )
        print(" OK")
        
        print("  ✓ Importing GroundednessLabeller, HiddenStateCollector & LiveGuardrailEnforcer...", end="")
        from services.critic import GroundednessLabeller, HiddenStateCollector, LiveGuardrailEnforcer
        print(" OK")
        
        print("  ✓ Importing Critic Models & Trainer API...", end="")
        from services.critic import BiLSTMCritic, CNNCritic, CriticDataset, train_critic_model
        print(" OK")
        
        print("\n✅ All modules imported successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_functionality():
    """Test basic functionality of key classes."""
    try:
        print("\nTesting basic functionality...")
        
        print("  ✓ Testing SFTExample...", end="")
        from services.training.sft_dataset import SFTExample
        ex = SFTExample(
            id="test_1",
            query="What is X?",
            answer="X is Y.",
        )
        assert ex.id == "test_1"
        print(" OK")
        
        print("  ✓ Testing SFTDatasetBuilder...", end="")
        from services.training.sft_dataset import SFTDatasetBuilder
        builder = SFTDatasetBuilder()
        chunks = [{"text": "Sample text", "chunk_id": "c1", "source_id": "doc1"}]
        examples = builder.create_from_chunks(chunks)
        assert len(examples) > 0
        print(" OK")
        
        print("  ✓ Testing DPOPreferenceGenerator...", end="")
        from services.training.dpo_generator import DPOPreferenceGenerator
        gen = DPOPreferenceGenerator()
        sft_ex = [{"query": "Q", "chosen": "A", "citations": []}]
        pairs = gen.generate_from_sft_examples(sft_ex)
        assert len(pairs) > 0
        print(" OK")
        
        print("  ✓ Testing GroundednessCase...", end="")
        from services.evaluation.groundedness_comparator import GroundednessCase
        case = GroundednessCase(
            id="c1",
            query="Q?",
            baseline_answer="Baseline",
            policy_answer="Policy",
            baseline_citations=[],
            policy_citations=[],
            baseline_guardrail={},
            policy_guardrail={},
        )
        assert case.id == "c1"
        print(" OK")

        print("  ✓ Testing GroundednessLabeller...", end="")
        from services.critic import GroundednessLabeller
        labeller = GroundednessLabeller()
        res = labeller.label_sentences("Word. Other.", "Word")
        assert len(res) == 2
        assert res[0][1] == 1
        assert res[1][1] == 0
        print(" OK")

        print("  ✓ Testing LiveGuardrailEnforcer...", end="")
        from services.critic import LiveGuardrailEnforcer
        enforcer = LiveGuardrailEnforcer()
        res = enforcer.score_and_enforce("q", "context word", "context word", "medical_prescription")
        assert res["critic_score"] == 0.0
        assert res["fallback_used"] is False
        print(" OK")

        print("  ✓ Testing BiLSTMCritic & CNNCritic...", end="")
        from services.critic import BiLSTMCritic, CNNCritic
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None:
            lstm = BiLSTMCritic(hidden_size=8, lstm_hidden=4)
            cnn = CNNCritic(hidden_size=8, cnn_channels=4)
            x = torch.randn((1, 3, 8))
            assert lstm(x).shape == (1, 1)
            assert cnn(x).shape == (1, 1)
            print(" OK (with Torch)")
        else:
            print(" SKIPPED (no Torch)")
        
        print("\n✅ All functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_imports() and test_basic_functionality()
    sys.exit(0 if success else 1)
