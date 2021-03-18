import os
import json
import torch
from torch.utils.data import Dataset

class IntentDataset(Dataset):
    def __init__(self, args, case, vocab, intent_label):
        self.data = json.load(open(os.path.join(args.data_dir, "{}.json".format(case))))
        self.vocab = vocab
        self.intent_label = intent_label
        self.label_intent = {idx: intent for intent, idx in self.intent_label.items()}
        self.max_seq_len = args.max_seq_len
        for i, data in enumerate(self.data):
            self.data[i]["text"] = data["text"].split()
            if "intent" in data:
                self.data[i]["label"] = self.intent_label[data["intent"]]

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
            if "label" in sample:
                labels.append(sample["label"])
            else:
                labels.append(-1)
        return ids, torch.LongTensor(self.vocab.encode_batch(texts, self.max_seq_len)), torch.LongTensor(text_lens), \
                torch.LongTensor(labels)

    @property
    def num_classes(self):
        return len(self.intent_label)

