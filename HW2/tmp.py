import json
import numpy as np
from transformers import BertTokenizerFast

from transformers import BertForQuestionAnswering

import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)
# Add match ID "HelloWorld" with no callback and one pattern
pattern = [{"IS_PUNCT": True}]
matcher.add("punc", [pattern])

doc = nlp("Hello, world! Hello world!")
matches = matcher(doc)
for match_id, start, end in matches:
    string_id = nlp.vocab.strings[match_id]  # Get string representation
    span = doc[start:end]  # The matched span
    print(match_id, string_id, start, end, span.text)
exit()

#with open("data/public.json") as f:
#    questions = json.load(f)
#print(np.percentile([len(q["question"]) for q in questions], 99))
#print(np.mean([q["answers"][0]["start"] for q in questions]))

from transformers import BertForQuestionAnswering
model = BertForQuestionAnswering.from_pretrained("bert-base-chinese")
exit()

with open("data/context.json") as f:
    context = json.load(f)

tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
print(tokenizer([list(i) for i in ["1", "012345"]], [list(i) for i in context[:2]], return_offsets_mapping=True, is_split_into_words=True))
#print(np.max([len(tokenizer(d)["input_ids"]) for d in context]))
#print(np.mean([len(tokenizer(d)["input_ids"]) for d in context]))
#print(np.mean([len(tokenizer(st)["input_ids"]) for d in context for st in d.strip().split('。')]))
#print(tokenizer.convert_ids_to_tokens([tokenizer(d.strip().split('。'), truncation=True, padding=True, max_length=512) for d in context][0]["input_ids"][0]))
