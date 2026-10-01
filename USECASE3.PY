# inference.py

import json
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

ADAPTER = "./grade_grievance_lora"


tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16
        if torch.cuda.is_available()
        else torch.float32
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER
)

model.eval()


def predict(question):

    prompt = f"""
### Instruction:
{question}

### Response:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    if torch.cuda.is_available():
        inputs = {
            k: v.cuda()
            for k, v in inputs.items()
        }

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=False
        )

    result = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    return result


question = (
    "I scored 35 in DL but expected 60. "
    "How do I challenge my grade?"
)

print(predict(question))
