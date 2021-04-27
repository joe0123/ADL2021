import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import unicodedata
import spacy
import bisect

def build_datasets(args, case):
    context_data = []
    for context in json.load(open(args.context_data, 'r')):
        context_data.append({"context": context, \
                            "context_": prevent_bert_clean_text(context)})
    
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

        self.ques_data = []
        for qi, q_data in enumerate(ques_data):
            d = dict()
            d["q_id"] = q_data["id"]
            d["question"] = q_data["question"]
            if case == "train":
                d["rel"], d["irrel"] = [], []
                for p_id in q_data["paragraphs"]:
                    p = context_data[p_id]["context"]
                    p_ = context_data[p_id]["context_"]
                    if p_id == q_data["relevant"]:
                        for a in q_data["answers"]:
                            start = a["start"]
                            end = start + len(a["text"]) - 1
                            assert a["text"] == p[start: end + 1]
                            d["rel"].append({"paragraph": p, "paragraph_": p_, \
                                            "answer": a["text"], "start": start, "end": end})
                    else:
                        d["irrel"].append({"paragraph": p, "paragraph_": p_})
            else:
                d["unknown"] = []
                for p_id in q_data["paragraphs"]:
                    p = context_data[p_id]["context"]
                    p_ = context_data[p_id]["context_"]
                    d["unknown"].append({"paragraph": p, "paragraph_": p_})
            self.ques_data.append(d)

        self.data = self.sample_data()

    def sample_data(self, irrel_ratio=None):    #TODO implement with irrel_ratio ratio
        data = []
        for qi, q_data in enumerate(self.ques_data):
            if self.case == "train":
                for r in q_data["rel"]:
                    data.append({"q_id": q_data["q_id"], "question": q_data["question"], \
                                "paragraph": r["paragraph"], "paragraph_": r["paragraph_"], "rel_label": 1, \
                                "answer": r["answer"], "start_label": r["start"], "end_label": r["end"]})
                for ir in q_data["irrel"]:
                    data.append({"q_id": q_data["q_id"], "question": q_data["question"], \
                                "paragraph": r["paragraph"], "paragraph_": r["paragraph_"], "rel_label": 0, \
                                "answer": '', "start_label": self.args.max_seq_len, "end_label": self.args.max_seq_len})
            else:
                for u in q_data["unknown"]:
                    data.append({"q_id": q_data["q_id"], "question": q_data["question"], \
                                "paragraph": r["paragraph"], "paragraph_": r["paragraph_"]})

        return data

    def collate_fn(self, samples):
        merged_samples = {"q_ids": [], "questions": [], "paragraphs": [], "paragraphs_": [],    \
                        "rel_labels": [], "answers": [], "start_labels": [], "end_labels": []}
        for sample in samples:
            merged_samples["q_ids"].append(sample["q_id"])
            merged_samples["questions"].append(sample["question"])
            merged_samples["paragraphs"].append(sample["paragraph"])
            merged_samples["paragraphs_"].append(sample["paragraph_"])
            if self.case == "train":
                merged_samples["answers"].append(sample["answer"])
                merged_samples["rel_labels"].append(sample["rel_label"])
                merged_samples["start_labels"].append(sample["start_label"])
                merged_samples["end_labels"].append(sample["end_label"])
                
        questions = [list(s) for s in merged_samples["questions"]]
        paragraphs_ = [list(s) for s in merged_samples["paragraphs_"]]
        batch_encodings = self.tokenizer(questions, paragraphs_,    \
                            padding=True, truncation=True, max_length=self.args.max_seq_len,  \
                            is_split_into_words=True, return_tensors="pt")
        merged_samples["text_ids"] = batch_encodings["input_ids"]
        merged_samples["type_ids"] = batch_encodings["token_type_ids"]
        merged_samples["mask_ids"] = batch_encodings["attention_mask"]

        for i, (type_ids, mask_ids, start_label, end_label) \
                    in enumerate(zip(merged_samples["type_ids"], merged_samples["mask_ids"], \
                                merged_samples["start_labels"], merged_samples["end_labels"])):
            base = ((1 - type_ids) * mask_ids).sum().item()
            start_label += base
            end_label += base
            # TODO ignored out of sentence in criterion or [SEP] here 
            #if start_label == -1 or start_label >= merged_samples["text_ids"].shape[1]:
            #    start_label = mask_ids.sum() - 1
            #if end_label == -1 or end_label >= merged_samples["text_ids"].shape[1]:
            #    end_label = mask_ids.sum() - 1
            merged_samples["start_labels"][i] = start_label
            merged_samples["end_labels"][i] = end_label

        #for text_ids, start_label, end_label in zip(merged_samples["text_ids"], 
        #                        merged_samples["start_labels"], merged_samples["end_labels"]):
        #    print(''.join(self.tokenizer.convert_ids_to_tokens(text_ids[start_label: end_label + 1])))
         
        merged_samples["rel_labels"] = torch.FloatTensor(merged_samples["rel_labels"])
        merged_samples["start_labels"] = torch.LongTensor(merged_samples["start_labels"])
        merged_samples["end_labels"] = torch.LongTensor(merged_samples["end_labels"])
        
        return merged_samples

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def prevent_bert_clean_text(text):
    """Prevent invalid character removal and whitespace cleanup in tokenizer."""
    
    def _is_control(char):
        """Checks whether `chars` is a control character."""
        if char == "\t" or char == "\n" or char == "\r":
            return False
        cat = unicodedata.category(char)
        if cat in ("Cc", "Cf"):
            return True
        return False
    
    def _is_whitespace(char):
        """Checks whether `chars` is a whitespace character."""
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False

    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or _is_control(char):
            output.append('_')
        if _is_whitespace(char):
            output.append('_')
        else:
            output.append(char)
    return ''.join(output)


