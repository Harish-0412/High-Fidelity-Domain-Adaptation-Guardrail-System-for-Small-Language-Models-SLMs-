import outlines
from pydantic import BaseModel
import json

class Test(BaseModel):
    name: str

print(dir(outlines.processors))
print(dir(outlines.json_schema))
print(dir(outlines.processors.base_logits_processor))
