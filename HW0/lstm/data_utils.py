import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from gensim.models import Word2Vec

class Dictionary:
    def __init__(self, w2v_path):
        self.word_index = {}
        self.index_word = []
        self.embed_matrix = []
        self.embed_model = Word2Vec.load(w2v_path)
        self.embed_dim = self.embed_model.vector_size
        self.add_zero_embed("<PAD>")
        self.add_random_embed("<UNK>")
        for i, w in enumerate(self.embed_model.wv.vocab):
            self.word_index[w] = len(self.word_index)
            self.index_word.append(w)
            self.embed_matrix.append(self.embed_model[w])
        self.embed_matrix = torch.tensor(self.embed_matrix)
        self.embed_matrix = (self.embed_matrix - torch.mean(self.embed_matrix, axis=0)) / torch.std(self.embed_matrix, axis=0)

    def add_zero_embed(self, w):
        v = np.zeros(self.embed_dim)
        self.word_index[w] = len(self.word_index)
        self.index_word.append(w)
        self.embed_matrix.append(v)

    def add_random_embed(self, w):
        v = np.random.uniform(size=self.embed_dim)
        self.word_index[w] = len(self.word_index)
        self.index_word.append(w)
        self.embed_matrix.append(v)
    
    def to_idx(self, st):
        result = []
        for s in st:
            if s in self.word_index:
                result.append(self.word_index[s])
            else:
                result.append(self.word_index["<UNK>"])
        
        return result

def make_sequence(sequence, max_seq_len, dtype="int64", padding="post", truncating="post", pad_value=0):
    if truncating == "prev":
        sequence = sequence[-max_seq_len:]
    else:
        sequence = sequence[:max_seq_len]

    text_ids = (np.ones(max_seq_len) * pad_value).astype(dtype)
    if padding == "post":
        text_ids[:len(sequence)] = sequence
    else:
        text_ids[-len(sequence):] = sequence
    
    return text_ids

class Tokenizer4LSTM:
    def __init__(self, dictionary, max_seq_len):
        self.dict = dictionary
        self.max_seq_len = max_seq_len
    
    def texts_to_sequence(self, texts, padding="post", truncating="post"):
        sequence = self.dict.to_idx(texts.split())
        return make_sequence(sequence, self.max_seq_len, padding=padding, truncating=truncating, \
                            pad_value=self.dict.word_index["<PAD>"])

class SentimentDataset(Dataset):
    def __init__(self, args, case, tokenizer):
        df = pd.read_csv(getattr(args, "{}_data".format(case)))
        ids, x, y = df.Id, df.text, df.Category

        self.data = []
        for raw_data in zip(ids, x, y):
            identity = raw_data[0]
            texts = raw_data[1].strip()
            polarity = float(raw_data[2])
            
            text_ids = tokenizer.texts_to_sequence(texts)
            
            data = {
                "id": identity, 
                "text_ids": text_ids,
                "polarity": polarity
            }
            self.data.append(data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

