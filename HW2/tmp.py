import json
import numpy as np
from transformers import BertTokenizerFast

from transformers import BertForQuestionAnswering


with open("data/train.json") as f:
    questions = json.load(f)
print(np.quantile([q["answers"][0]["start"] for q in questions], 0.1))
exit()
with open("data/context.json") as f:
    context = json.load(f)

tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
print(tokenizer(context[0], return_offsets_mapping=True))
#print(np.max([len(tokenizer(d)["input_ids"]) for d in context]))
#print(np.mean([len(tokenizer(d)["input_ids"]) for d in context]))
#print(np.mean([len(tokenizer(st)["input_ids"]) for d in context for st in d.strip().split('。')]))
#print(tokenizer.convert_ids_to_tokens([tokenizer(d.strip().split('。'), truncation=True, padding=True, max_length=512) for d in context][0]["input_ids"][0]))
