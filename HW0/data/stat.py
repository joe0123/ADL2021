import pandas as pd
import numpy as np
from transformers import BertTokenizer

print("[For LSTM]")
train_df = pd.read_csv("train.csv")
train_lens = [len(st.split()) for st in train_df.text]
valid_df = pd.read_csv("dev.csv")
valid_lens = [len(st.split()) for st in valid_df.text]
test_df = pd.read_csv("test.csv")
test_lens = [len(st.split()) for st in test_df.text]
print("PR95 of train text length: {}".format(np.percentile(train_lens, 95)))
print("PR95 of valid text length: {}".format(np.percentile(valid_lens, 95)))
print("PR95 of test text length: {}".format(np.percentile(test_lens, 95)))

print("[For Bert]")
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
train_df = pd.read_csv("train.csv")
train_lens = [len(tokenizer.tokenize(st.replace(' ', ''))) for st in train_df.text]
valid_df = pd.read_csv("dev.csv")
valid_lens = [len(tokenizer.tokenize(st.replace(' ', ''))) for st in valid_df.text]
test_df = pd.read_csv("test.csv")
test_lens = [len(tokenizer.tokenize(st.replace(' ', ''))) for st in test_df.text]
print("PR95 of train text length: {}".format(np.percentile(train_lens, 95)))
print("PR95 of valid text length: {}".format(np.percentile(valid_lens, 95)))
print("PR95 of test text length: {}".format(np.percentile(test_lens, 95)))

