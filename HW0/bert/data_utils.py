import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def make_sequence(sequence, max_seq_len, cls_id, sep_id, dtype="int64", padding="post", truncating="post", pad_value=0):
    max_text_len = max_seq_len - 2
    if truncating == "prev":
        sequence = sequence[-max_text_len:]
    else:
        sequence = sequence[:max_text_len]

    sequence.insert(0, cls_id)
    sequence.append(sep_id)

    text_ids = (np.ones(max_seq_len) * pad_value).astype(dtype)
    mask_ids = (np.ones(max_seq_len) * pad_value).astype(dtype)
    if padding == "post":
        text_ids[:len(sequence)] = sequence
        mask_ids[:len(sequence)] = 1
    else:
        text_ids[-len(sequence):] = sequence
        mask_ids[-len(sequence):] = 1
    
    return text_ids, mask_ids

class Tokenizer4Bert:
    def __init__(self, max_seq_len, tokenizer):
        self.tokenizer = tokenizer 
        self.max_seq_len = max_seq_len

    def texts_to_sequence(self, texts, padding="post", truncating="post"):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(texts))
        if len(sequence) == 0:
            sequence = [0]
        
        return make_sequence(sequence, self.max_seq_len, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, \
                                padding=padding, truncating=truncating)

class SentimentDataset(Dataset):
    def __init__(self, args, case, tokenizer):
        df = pd.read_csv(getattr(args, "{}_data".format(case)))
        ids, x, y = df.Id, df.text, df.Category

        self.data = []
        for raw_data in zip(ids, x, y):
            identity = raw_data[0]
            texts = raw_data[1].strip().replace(' ', '')
            polarity = float(raw_data[2])

            bert_text_ids, bert_mask_ids = tokenizer.texts_to_sequence(texts)
            
            data = {
                "id": identity, 
                "bert_text_ids": bert_text_ids,
                "bert_mask_ids": bert_mask_ids,
                "polarity": polarity
            }
            self.data.append(data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

