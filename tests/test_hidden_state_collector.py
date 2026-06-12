from __future__ import annotations

import json
import tempfile
import pytest
torch = pytest.importorskip("torch")
from pathlib import Path

from services.critic.collector import GroundednessLabeller, HiddenStateCollector


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------

class MockModelConfig:
    def __init__(self, num_layers=4, hidden_size=8):
        self.num_hidden_layers = num_layers
        self.hidden_size = hidden_size


class MockModel:
    def __init__(self, num_layers=4, hidden_size=8):
        self.config = MockModelConfig(num_layers, hidden_size)
        self.device = "cpu"

    def eval(self):
        pass

    def generate(self, input_ids, output_hidden_states=False, return_dict_in_generate=False, max_new_tokens=4, **kwargs):
        prompt_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        hidden_size = self.config.hidden_size
        num_layers = self.config.num_hidden_layers

        # Generated IDs: 100, 101, 102, 103
        generated_ids = torch.arange(100, 100 + max_new_tokens, dtype=torch.long)
        sequences = torch.cat([input_ids[0], generated_ids]).unsqueeze(0)

        hidden_states = []
        for t in range(max_new_tokens):
            step_states = []
            seq_len = prompt_len if t == 0 else 1
            for l in range(num_layers + 1):
                # Value matches layer and step to verify indexing correctness
                val = float(t * 10 + l)
                t_state = torch.full((batch_size, seq_len, hidden_size), val, dtype=torch.float)
                step_states.append(t_state)
            hidden_states.append(tuple(step_states))

        class GenerateOutput:
            def __init__(self, seqs, states):
                self.sequences = seqs
                self.hidden_states = tuple(states)

        return GenerateOutput(sequences, hidden_states)


class MockTokenizer:
    def __init__(self):
        self.pad_token = "[PAD]"
        self.eos_token = "[EOS]"
        self.padding_side = "left"

    def __call__(self, text, return_tensors=None, **kwargs):
        input_ids = torch.arange(1, 7, dtype=torch.long).unsqueeze(0)  # Length 6 prompt
        class MockBatchEncoding(dict):
            def to(self, device):
                return self
        return MockBatchEncoding({"input_ids": input_ids})

    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, (list, tuple, torch.Tensor)):
            decoded = []
            for tid in token_ids:
                if isinstance(tid, torch.Tensor):
                    tid = tid.item()
                if tid >= 100:
                    step = int(tid - 100)
                    decoded.append(f"word{step}")
                else:
                    decoded.append(f"p{tid}")
            
            # Form clean sentences: "word0 word1." and "word2 word3."
            text_tokens = []
            for i, word in enumerate(decoded):
                if word.startswith("word"):
                    step = int(word[4:])
                    if step % 2 == 1:
                        text_tokens.append(f"{word}.")
                    else:
                        text_tokens.append(word)
                else:
                    text_tokens.append(word)
            return " ".join(text_tokens)
        else:
            # Single ID decode
            if isinstance(token_ids, torch.Tensor):
                tid = token_ids.item()
            else:
                tid = int(token_ids)

            if tid >= 100:
                step = int(tid - 100)
                if step % 2 == 1:
                    return f"word{step}."
                else:
                    return f"word{step}"
            return f"p{tid}"


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_groundedness_labeller():
    labeller = GroundednessLabeller(threshold=0.5)
    
    # Grounded sentence matches source text terms
    answer = "The dosage of albuterol is two inhalations. Take it every four hours."
    source = "Albuterol sulfate dosage: two inhalations every 4 to 6 hours."
    
    results = labeller.label_sentences(answer, source)
    
    assert len(results) == 2
    assert results[0][0] == "The dosage of albuterol is two inhalations."
    # Matches multiple words: dosage, albuterol, two, inhalations
    assert results[0][1] == 1 
    
    # Second sentence has no overlap with source
    assert results[1][0] == "Take it every four hours."
    assert results[1][1] == 0


def test_collector_runs_generation_and_slices_layers():
    model = MockModel(num_layers=8, hidden_size=4)
    tokenizer = MockTokenizer()
    collector = HiddenStateCollector(model=model, tokenizer=tokenizer, device="cpu")

    # Run collection with specific middle-to-late layers [6, 7, 8]
    records = collector.collect_from_query(
        query="test query",
        source_chunk="word0 word1",  # Grounded sentence text
        source_id="source_doc",
        layer_indices=[6, 7, 8],
        max_new_tokens=4,
    )

    # Generated tokens are word0, word1, word2, word3.
    # Four words, 3 layers per word = 12 total entries.
    assert len(records) == 12

    # Check first record structure
    first = records[0]
    assert first["token"] in ("word0", "word0.")
    assert first["layer_index"] == 6
    assert first["source_id"] == "source_doc"
    assert first["grounded_label"] == 1  # Matches context: "word0 word1"
    assert len(first["hidden_state"]) == 4

    # Verify KV caching indexing:
    # step_idx 0 (word0), layer 6 should have value 0*10 + 6 = 6.0
    assert first["hidden_state"][0] == 6.0

    # step_idx 1 (word1), layer 7 should have value 1*10 + 7 = 17.0
    word1_layer7 = [r for r in records if r["token"] in ("word1", "word1.") and r["layer_index"] == 7][0]
    assert word1_layer7["hidden_state"][0] == 17.0

    # step_idx 2 (word2), layer 8 should have value 2*10 + 8 = 28.0
    word2_layer8 = [r for r in records if r["token"] in ("word2", "word2.") and r["layer_index"] == 8][0]
    assert word2_layer8["hidden_state"][0] == 28.0


def test_collector_groundedness_labels():
    model = MockModel(num_layers=4, hidden_size=4)
    tokenizer = MockTokenizer()
    collector = HiddenStateCollector(model=model, tokenizer=tokenizer, device="cpu")

    # source_chunk contains "word0 word1" but not "word2 word3"
    records = collector.collect_from_query(
        query="test",
        source_chunk="word0 word1",
        source_id="src",
        layer_indices=[3],
    )

    # word0, word1 should be labeled 1 (grounded)
    word0_records = [r for r in records if r["token"] in ("word0", "word0.")]
    word2_records = [r for r in records if r["token"] in ("word2", "word2.")]

    assert all(r["grounded_label"] == 1 for r in word0_records)
    # word2 should be labeled 0 (unsupported/hallucinated)
    assert all(r["grounded_label"] == 0 for r in word2_records)
