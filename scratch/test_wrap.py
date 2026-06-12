from pydantic import BaseModel
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from outlines import from_transformers
from outlines.generator import get_json_schema_logits_processor

class Test(BaseModel):
    name: str

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

outlines_model = from_transformers(hf_model, tokenizer)
lp = get_json_schema_logits_processor(None, outlines_model, json.dumps(Test.model_json_schema()))
print("LogitsProcessor Type:", type(lp))
