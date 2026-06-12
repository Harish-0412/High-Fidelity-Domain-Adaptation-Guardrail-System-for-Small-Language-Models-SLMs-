from pydantic import BaseModel
import json
import torch
from transformers import AutoTokenizer

class Test(BaseModel):
    name: str

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B") # tiny model for test
try:
    from outlines.models.transformers import TransformerTokenizer
    from outlines.fsm.guide import RegexGuide
    from outlines.fsm.json_schema import build_regex_from_schema
    from outlines.processors import OutlinesLogitsProcessor

    outlines_tok = TransformerTokenizer(tokenizer)
    regex_str = build_regex_from_schema(json.dumps(Test.model_json_schema()))
    guide = RegexGuide(regex_str, outlines_tok)
    lp = OutlinesLogitsProcessor(tokenizer, guide)
    print("Success building LogitsProcessor from Outlines!")
except Exception as e:
    import traceback
    traceback.print_exc()

# Let's also check if outlines provides a high-level wrapper
try:
    from outlines import generate
    print(dir(generate))
except Exception as e:
    print(e)
