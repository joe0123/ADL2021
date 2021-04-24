import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import unicodedata

def build_datasets(args, case):
    context_data = [prevent_clean_text(d) for d in json.load(open(args.context_data, 'r'))]
    ques_data = json.load(open(getattr(args, "{}_data".format(case)), 'r'))
    if case == "train" and args.eval_ratio > 0:
        ques_data = np.array(ques_data)
        ids = np.random.permutation(ques_data.shape[0])
        cut = int(ques_data.shape[0] * args.eval_ratio)
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
        
        # TODO 改在這打散句子
        context_data_ = [prevent_clean_text(context) for context in context_data]
        self.ques_data = []
        for qi, q_data in enumerate(ques_data):
            d = dict()
            d["q_id"] = q_data["id"]
            d["question"] = q_data["question"]
            if case == "train":
                d["rel"], d["irrel"] = [], []
                for p_id in q_data["paragraphs"]:
                    p, p_ = context_data[p_id], context_data_[p_id]
                    if p_id == q_data["relevant"]:
                        for a in q_data["answers"]:
                            d["rel"].append({"paragraph": p, "paragraph_": p_, \
                                    "answer": a["text"], "start": a["start"], "end": a["start"] + len(a["text"]) - 1})
                    else:
                        d["irrel"].append({"paragraph": p, "paragraph_": p_})
            else:
                d["unknown"] = [{"paragraph": context_data[p_id], "paragraph_": context_data_[p_id]} \
                                        for p_id in q_data["paragraphs"]]
            self.ques_data.append(d)

        self.data = self.sample_data()

    def sample_data(self, irrel_ratio=None):    #TODO implement with irrel_ratio ratio
        data = []
        for qi, q_data in enumerate(self.ques_data):
            if self.case == "train":
                for r in q_data["rel"]:
                    data.append({"q_id": q_data["q_id"], "question": q_data["question"], \
                                "paragraph": r["paragraph"], "paragraph_": r["paragraph_"], \
                                "rel_label": 1, "start_label": r["start"], "end_label": r["end"]})
                for ir in q_data["irrel"]:
                    data.append({"q_id": q_data["q_id"], "question": q_data["question"], \
                                "paragraph": r["paragraph"], "paragraph_": r["paragraph_"], \
                                "rel_label": 0, "start_label": -1, "end_label": -1})
            else:
                for u in q_data["unknown"]:
                    data.append({"q_id": q_data["q_id"], "question": q_data["question"], \
                                "paragraph": r["paragraph"], "paragraph_": r["paragraph_"]})

        return data

    def collate_fn(self, samples):
        all_q_ids, all_questions, all_paragraphs, all_paragraphs_ = [], [], [], []
        all_rel_labels, all_start_labels, all_end_labels = [], [], []
        for sample in samples:
            all_q_ids.append(sample["q_id"])
            all_questions.append(list(sample["question"]))
            all_paragraphs.append(list(sample["paragraph"]))
            all_paragraphs_.append(list(sample["paragraph_"]))
            if self.case == "train":
                all_rel_labels.append(sample["rel_label"])
                all_start_labels.append(sample["start_label"])
                all_end_labels.append(sample["end_label"])
        
        batch_encodings = self.tokenizer(all_questions, all_paragraphs_, padding=True, truncation=True, \
                            is_split_into_words=True, max_length=self.args.max_seq_len, return_tensors="pt")
        all_text_ids = batch_encodings["input_ids"]
        all_type_ids = batch_encodings["token_type_ids"]
        all_mask_ids = batch_encodings["attention_mask"]

        for type_ids, mask_ids, start_label, end_label \
                    in zip(all_type_ids, all_mask_ids, all_start_labels, all_end_labels):
            base = ((1 - type_ids) * mask_ids).sum().item()
            start_label += (base if start_label != -1 else 0)
            end_label += (base if end_label != -1 else 0)
            if start_label == -1 or start_label >= all_text_ids.shape[1]:
                start_label = mask_ids.sum() - 1
            if end_label == -1 or end_label >= all_text_ids.shape[1]:
                end_label = mask_ids.sum() - 1
            
            #print(q_id, ''.join(self.tokenizer.convert_ids_to_tokens(text_ids[start_label: end_label + 1])))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def prevent_clean_text(text):
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


