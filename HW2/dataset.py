import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import bisect

def build_datasets(args, case):
    context_data = json.load(open(args.context_data, 'r'))
    ques_data = json.load(open(getattr(args, "{}_data".format(case)), 'r'))
    if case == "train" and args.eval_ratio > 0:
        ques_data = np.array(ques_data)
        ids = np.random.permutation(ques_data.shape[0])
        cut = int(ques_data.shape[0] * args.eval_ratio)
        train_ques_data = ques_data[ids[cut:]].tolist()
        eval_ques_data = ques_data[ids[:cut]].tolist()
        return QADataset(args, context_data, train_ques_data), QADataset(args, context_data, eval_ques_data)
    else:
        return QADataset(args, context_data, ques_data)



class QADataset(Dataset):
    def __init__(self, args, context_data, ques_data):
        self.args = args
        self.tokenizer = args.bert_tokenizer
        self.ques_data = ques_data
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
            for paragraph, rel_label, start_labels, end_labels in zip(sample["paragraphs"], sample["rel_labels"], \
                                                                    sample["start_labels"], sample["end_labels"]):
                for start_label, end_label in zip(start_labels, end_labels):
                    all_questions.append(list(sample["question"]))
                    all_paragraphs.append(list(paragraph))
                    all_rel_labels.append(rel_label)
                    all_start_labels.append(start_label)
                    all_end_labels.append(end_label)
        
        batch_encodings = self.tokenizer(all_questions, all_paragraphs, padding=True, truncation=True, \
                            is_split_into_words=True, max_length=self.args.max_seq_len, \
                            return_offsets_mapping=True, return_tensors="pt")
        all_text_ids = batch_encodings["input_ids"]
        all_type_ids = batch_encodings["token_type_ids"]
        all_mask_ids = batch_encodings["attention_mask"]
        all_offset_mappings = batch_encodings["offset_mapping"].tolist()

        for q, text_ids, p, type_ids, mask_ids, offset_mappings, start_label, end_label \
                in zip(all_questions, all_text_ids, all_paragraphs, all_type_ids, all_mask_ids, all_offset_mappings, all_start_labels, all_end_labels):
            low = torch.sum((1 - type_ids) * mask_ids).item()
            high = torch.sum(mask_ids).item() - 1
            if start_label != -1:
                new_start_label = bisect.bisect_left([i[0] for i in offset_mappings], start_label, low, high)
                if new_start_label >= len(offset_mappings) - 1:
                    new_start_label = -1    # TODO remove out of bound or not
                else:
                    assert offset_mappings[new_start_label][0] == start_label, print(start_label, new_start_label)
            if end_label != -1:
                new_end_label = bisect.bisect_left([i[1] - 1 for i in offset_mappings], end_label, low, high)
                if new_end_label >= len(offset_mappings) - 1:
                    new_end_label = -1    # TODO remove out of bound or not
                else:
                    assert offset_mappings[new_end_label][1] - 1 == end_label, print(q, p[start_label: end_label + 1], offset_mappings, end_label, new_end_label)



    def __getitem__(self, index):
        return self.ques_data[index]

    def __len__(self):
        return len(self.ques_data)
