import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

def build_dataset():


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
        all_questions, all_paragraphs, all_rel_labels, all_start_labels, all_end_labels = [], [], [], [], []
        for sample in samples:
            for paragraph, rel_label, start_labels, end_labels in zip(sample["paragraphs"], sample["rel_labels"], sample["start_labels", sample["end_labels"]]):
                for start, end in zip(start_labels, end_labels):
                    all_questions.append(sample["question"])
                    all_paragraphs.append(sample["paragraphs"])
                    all_rel_labels.append(sample["rel_labels"])
                    all_start_labels.append(sample["start_labels"])
                    all_end_labels.append(sample["end_labels"])
        
        batch_encodings = self.tokenizer(all_questions, all_paragraphs, padding=True, truncation=True, \
                        max_length=self.args.max_seq_len, return_offsets_mapping=True)
        all_text_ids = batch_encodings["input_ids"]
        all_type_ids = batch_encodings["token_type_ids"]
        all_mask_ids = batch_encodings["attention_masks"]
        all_offset_mappings = batch_encodings["offset_mapping"]

        for type_ids, mask_ids, offset_mappings, start_labels, end_labels
        

    def __getitem__(self, index):
        return self.ques_data[index]

    def __len__(self):
        return len(self.ques_data)
