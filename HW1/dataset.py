import os
import json
import pickle
import torch
from torch.utils.data import Dataset

from data_utils import pad_to_len

class IntentDataset(Dataset):
    def __init__(self, args, cases):
        self.data = []
        for case in cases:
            self.data += json.load(open(os.path.join(args.data_dir, "{}.json".format(case))))
        self.vocab = pickle.load(open(os.path.join(args.cache_dir, "vocab.pkl"), "rb"))
        self.label2id = json.load(open(os.path.join(args.cache_dir, "intent2idx.json"), 'r'))
        self.id2label = {idx: intent for intent, idx in self.label2id.items()}
        self.max_seq_len = args.max_seq_len
        for i, data in enumerate(self.data):
            self.data[i]["text"] = data["text"].split()
            if "intent" in data:
                self.data[i]["label_id"] = self.label2id[data["intent"]]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def collate_fn(self, samples):
        ids, texts, text_lens, labels = [], [], [], []
        for sample in samples:
            ids.append(sample["id"])
            texts.append(sample["text"])
            text_lens.append(len(sample["text"]))
            if "label_id" in sample:
                labels.append(sample["label_id"])
            else:
                labels.append(-1)
        return ids, torch.LongTensor(self.vocab.encode_batch(texts, self.max_seq_len)), torch.LongTensor(text_lens), \
                torch.LongTensor(labels)

    @property
    def num_classes(self):
        return len(self.label2id)


class SlotDataset(Dataset):
    def __init__(self, args, cases):
        self.data = []
        for case in cases:
            self.data += json.load(open(os.path.join(args.data_dir, "{}.json".format(case))))
        self.vocab = pickle.load(open(os.path.join(args.cache_dir, "vocab.pkl"), "rb"))
        self.label2id = json.load(open(os.path.join(args.cache_dir, "tag2idx.json"), 'r'))
        self.id2label = {idx: tag for tag, idx in self.label2id.items()}
        self.max_seq_len = args.max_seq_len
        for i, data in enumerate(self.data):
            self.data[i]["text"] = data["tokens"]
            if "tags" in data:
                self.data[i]["label_id"] = [self.label2id[t] for t in data["tags"]]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def collate_fn(self, samples):
        ids, texts, text_lens, label_ids = [], [], [], []
        for sample in samples:
            ids.append(sample["id"])
            texts.append(sample["text"])
            text_lens.append(len(sample["text"]))
            if "label_id" in sample:
                labels.append(sample["label_id"])
            else:
                labels.append(-1)
        text_ids = torch.LongTensor(self.vocab.encode_batch(texts, self.max_seq_len))
        text_lens = torch.LongTensor(text_lens)
        labels = torch.LongTensor(pad_to_len(labels, texts.shape[0], self.label2id["[PAD]"]))
        return ids, texts, text_lens, labels

    @property
    def num_classes(self):
        return len(self.label2id)
    

