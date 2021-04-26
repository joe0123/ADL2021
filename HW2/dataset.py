import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import unicodedata
import spacy
import bisect
from joblib import Parallel, delayed

def build_datasets(args, case):
    context_data = json.load(open(args.context_data, 'r'))
    
    ques_data = json.load(open(getattr(args, "{}_data".format(case)), 'r'))
    if case == "train" and args.valid_ratio > 0:
        ques_data = np.array(ques_data)
        ids = np.random.permutation(ques_data.shape[0])
        cut = int(ques_data.shape[0] * args.valid_ratio)
        train_ques_data = ques_data[ids[cut:]].tolist()
        eval_ques_data = ques_data[ids[:cut]].tolist()
        return QADataset(args, context_data, train_ques_data, case), QADataset(args, context_data, eval_ques_data, case)
    else:
        return QADataset(args, context_data, ques_data, case)

class QADataset(Dataset):
    def __init__(self, args, context_data, ques_data, case):
        self.args = args
        self.case = case
        self.tokenizer = args.bert_tokenizer
        self.context_data = context_data
        self.case = case

        self.ques_data = Parallel(n_jobs=4)(delayed(self.make_ques_data)(q_data) for q_data in ques_data)
        
        self.data = self.sample_data()
    
    def make_ques_data(self, q_data):
        d = dict()
        d["q_id"] = q_data["id"]
        q = q_data["question"]
        d["rel"], d["irrel"], d["unknown"] = [], [], []
        for p_id in q_data["paragraphs"]:
            p = self.context_data[p_id]
            e = self.tokenizer(q, p, return_tensors="pt", return_offsets_mapping=True,    \
                padding="max_length", truncation="only_second", max_length=self.args.max_seq_len,   \
                return_overflowing_tokens=True)
            for text_ids, type_ids, mask_ids, offset_map \
                    in zip(e["input_ids"], e["token_type_ids"], e["attention_mask"], e["offset_mapping"]):
                if self.case == "train":
                    if p_id == q_data["relevant"]:
                        for a in q_data["answers"]:
                            start = a["start"]
                            end = start + len(a["text"]) - 1
                            start_label = ((1 - type_ids) * mask_ids).sum().item()
                            end_label = len(offset_map) - ((1 - type_ids) * (1 - mask_ids)).sum().item() - 2
                            if not(offset_map[start_label][0] <= start and offset_map[end_label][1] > end):
                                d["irrel"].append({"paragraph": p, "text_ids": text_ids, \
                                "type_ids": type_ids, "mask_ids": mask_ids, "offset_map": offset_map, \
                                "answer": '', "start_label": self.args.max_seq_len, "end_label": self.args.max_seq_len})
                            else:
                                while start_label < len(offset_map) and offset_map[start_label][0] <= start:
                                    start_label += 1
                                start_label -= 1
                                # TODO end label correct?
                                while offset_map[end_label][1] - 1 >= end:
                                    end_label -= 1
                                end_label += 1
                                if a["text"] != p[offset_map[start_label][0]: offset_map[end_label][1]]:
                                    print(a["text"], p[offset_map[start_label][0]: offset_map[end_label][1]])
                                d["rel"].append({"paragraph": p, "text_ids": text_ids, \
                                    "type_ids": type_ids, "mask_ids": mask_ids, "offset_map": offset_map, \
                                    "answer": a["text"], "start_label": start_label, "end_label": end_label})
                    else:
                        d["irrel"].append({"paragraph": p, "text_ids": text_ids, \
                            "type_ids": type_ids, "mask_ids": mask_ids, "offset_map": offset_map, \
                            "answer": '', "start_label": self.args.max_seq_len, "end_label": self.args.max_seq_len})
                else:
                    d["unknown"].append({"paragraph": p, "text_ids": text_ids, \
                        "type_ids": type_ids, "mask_ids": mask_ids, "offset_map": offset_map})
        return d

    def sample_data(self, irrel_ratio=None):    #TODO implement with irrel_ratio ratio
        data = []
        for qi, q_data in enumerate(self.ques_data):
            if self.case == "train":
                for r in q_data["rel"]:
                    data.append({"q_id": q_data["q_id"], "rel_label": 1, **r})
                for ir in q_data["irrel"]:
                    data.append({"q_id": q_data["q_id"], "rel_label": 0, **r})
            else:
                for u in q_data["unknown"]:
                    data.append({"q_id": q_data["q_id"], **r})

        return data

    
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


