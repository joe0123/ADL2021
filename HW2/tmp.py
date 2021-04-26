import json
import numpy as np
from transformers import BertTokenizerFast

with open("data/context.json") as f:
    context = json.load(f)

tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
output = tokenizer(["1", "012345"], context[:2], padding="max_length", max_length=512, return_overflowing_tokens=True, truncation="only_second", return_offsets_mapping=True, stride=128)
print(output)
#print(np.max([len(tokenizer(d)["input_ids"]) for d in context]))
#print(np.mean([len(tokenizer(d)["input_ids"]) for d in context]))
#print(np.mean([len(tokenizer(st)["input_ids"]) for d in context for st in d.strip().split('。')]))
#print(tokenizer.convert_ids_to_tokens([tokenizer(d.strip().split('。'), truncation=True, padding=True, max_length=512) for d in context][0]["input_ids"][0]))
