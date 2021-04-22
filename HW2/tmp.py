import json
import numpy as np
from transformers import BertTokenizerFast

with open("data/context.json") as f:
    context = json.load(f)

tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
print([tokenizer(d.strip().split('。'), truncation=True, padding=True, max_length=512) for d in context][0])
