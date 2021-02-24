import pandas as pd
import numpy as np
import re
from transformers import BertTokenizer

rule_sub = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5]")
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
train_df = pd.read_csv("train.csv")
train_lens = [len(tokenizer.tokenize(rule_sub.sub('', st))) for st in train_df.text]
valid_df = pd.read_csv("dev.csv")
valid_lens = [len(tokenizer.tokenize(rule_sub.sub('', st))) for st in valid_df.text]
test_df = pd.read_csv("test.csv")
test_lens = [len(tokenizer.tokenize(rule_sub.sub('', st))) for st in test_df.text]
print("PR98 of train text length: {}".format(np.percentile(train_lens, 95)))
print("PR98 of valid text length: {}".format(np.percentile(valid_lens, 95)))
print("PR98 of test text length: {}".format(np.percentile(test_lens, 95)))
