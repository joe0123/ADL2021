import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import unicodedata

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
            output.append('|')
        if _is_whitespace(char):
            output.append('|')
        else:
            output.append(char)
    return ''.join(output)


class QADataset(Dataset):
    def __init__(self, args, context_data, ques_data):
        self.args = args
        self.tokenizer = args.bert_tokenizer
        self.tokenizer.do_lower_case = True
        print(self.tokenizer.__dict__)
        self.ques_data = ques_data
        for qi, q_data in enumerate(self.ques_data):
            self.ques_data[qi]["rel_labels"] = [1 if id_ == q_data["relevant"] else 0 for id_ in q_data["paragraphs"]]
            self.ques_data[qi]["paragraphs"] = [prevent_clean_text(context_data[id_]) for id_ in q_data["paragraphs"]]
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

            # TODO 改在這打散句子
                        
    
    def collate_fn(self, samples):
        all_ids, all_questions, all_paragraphs, all_rel_labels, all_start_labels, all_end_labels = [], [], [], [], [], []
        for sample in samples:
            for paragraph, rel_label, start_labels, end_labels in zip(sample["paragraphs"], sample["rel_labels"], \
                                                                    sample["start_labels"], sample["end_labels"]):
                for start_label, end_label in zip(start_labels, end_labels):
                    all_ids.append(sample["id"])
                    all_questions.append(list(sample["question"]))
                    all_paragraphs.append(list(paragraph))
                    all_rel_labels.append(rel_label)
                    all_start_labels.append(start_label)
                    all_end_labels.append(end_label)
        
        batch_encodings = self.tokenizer(all_questions, all_paragraphs, padding=True, truncation=True, \
                            is_split_into_words=True, max_length=self.args.max_seq_len, return_tensors="pt")
        all_text_ids = batch_encodings["input_ids"]
        all_type_ids = batch_encodings["token_type_ids"]
        all_mask_ids = batch_encodings["attention_mask"]

        for id_, p, q, text_ids, type_ids, mask_ids, start_label, end_label \
                in zip(all_ids, all_paragraphs, all_questions, all_text_ids, all_type_ids, all_mask_ids, all_start_labels, all_end_labels):
            base = ((1 - type_ids) * mask_ids).sum().item()
            start_label += (base if start_label != -1 else 0)
            end_label += (base if end_label != -1 else 0)
            if start_label == -1 or start_label >= all_text_ids.shape[1]:
                start_label = mask_ids.sum() - 1
            if end_label == -1 or end_label >= all_text_ids.shape[1]:
                end_label = mask_ids.sum() - 1
            
            if id_ == "eb00030d925a8018ba074e88112071af" and start_label < mask_ids.sum() - 1:
                print(type_ids, mask_ids)
                print(p[467:469])
                print(start_label, end_label, base)
                print([(i, j) for i, j in zip(p, [self.tokenizer.convert_ids_to_tokens(t.item()) for t in text_ids[13:-1]])])
                print(self.tokenizer.convert_ids_to_tokens(text_ids[start_label: end_label + 1]))
                exit()


    def __getitem__(self, index):
        return self.ques_data[index]

    def __len__(self):
        return len(self.ques_data)
