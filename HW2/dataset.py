import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

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
