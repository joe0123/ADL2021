import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import re

class QADataset(Dataset):
    def __init__(self, args, case):
        #self.tokenizer = args.tokenizer

        context_data = json.load(open(args.context_data, 'r'))
        self.ques_data = json.load(open(getattr(args, "{}_data".format(case)), 'r'))
        
        for qi, q_data in enumerate(self.ques_data):
            self.ques_data[qi]["rel_labels"] = [1 if id_ == q_data["relevant"] else 0 for id_ in q_data["paragraphs"]]
            self.ques_data[qi]["paragraphs"] = [context_data[id_] for id_ in q_data["paragraphs"]]
            self.ques_data[qi]["start_labels"] = [[a["start"] for a in q_data["answers"]] if rel == 1 \
                                                else [-1] for rel in q_data["rel_labels"]]
            self.ques_data[qi]["end_labels"] = [[a["start"] + len(a["text"]) - 1 for a in q_data["answers"]] if rel == 1 \
                                                else [-1] for rel in q_data["rel_labels"]]
            
            #found = False
            #for starts, ends, paragraph in zip(q_data["start_labels"], q_data["end_labels"], q_data["paragraphs"]):
            #    if starts == [-1] and ends == [-1]:
            #        continue
            #    for start, end, a in zip(starts, ends, q_data["answers"]):
            #        print(a["text"], paragraph[start: end + 1])
            #        assert a["text"] == paragraph[start: end + 1]
            #        found = True
            #assert found
                        
    
    def collate_fn(self, samples):
        text_ids, rel_labels, start_labels, end_labels = [], [], [], []
        for sample in samples:
            question = samples["question"]
            for 


    def __getitem__(self, index):
        return self.ques_data[index]

    def __len__(self):
        return len(self.ques_data)
