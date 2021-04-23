import json
import numpy as np
from transformers import BertTokenizerFast

from transformers import BertForQuestionAnswering


#with open("data/public.json") as f:
#    questions = json.load(f)
#print(np.percentile([len(q["question"]) for q in questions], 99))
#print(np.mean([q["answers"][0]["start"] for q in questions]))


with open("data/context.json") as f:
    context = json.load(f)

tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
print(tokenizer(["1", "012345"], context[:2], return_offsets_mapping=True))
#print(np.max([len(tokenizer(d)["input_ids"]) for d in context]))
#print(np.mean([len(tokenizer(d)["input_ids"]) for d in context]))
#print(np.mean([len(tokenizer(st)["input_ids"]) for d in context for st in d.strip().split('。')]))
#print(tokenizer.convert_ids_to_tokens([tokenizer(d.strip().split('。'), truncation=True, padding=True, max_length=512) for d in context][0]["input_ids"][0]))
