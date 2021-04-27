import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


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
        self.tokenizer = args.bert_tokenizer
        
        all_q_ids, all_questions, all_paragraphs, all_rels, all_answers = [], [], [], [], []
        for q_data in ques_data:
            for p_id in q_data["paragraphs"]:
                all_q_ids.append(q_data["id"])
                all_questions.append(q_data["question"])
                all_paragraphs.append(context_data[p_id])
                if case == "train":
                    if p_id == q_data["relevant"]:
                        all_rels.append(1)
                        all_answers.append(q_data["answers"])
                    else:
                        all_rels.append(0)
                        all_answers.append(None)
                else:
                    all_rels.append(-1)
                    all_answers.append(None)
        
        all_encodings = self.tokenizer(all_questions, all_paragraphs, return_tensors="pt", return_offsets_mapping=True, \
                            padding="max_length", truncation="only_second", max_length=self.args.max_seq_len,   \
                            stride=self.args.stride, return_overflowing_tokens=True)
        print(len(all_encodings["overflow_to_sample_mapping"]))
        collector = dict()
        for ei, i in enumerate(all_encodings["overflow_to_sample_mapping"].tolist()):
            q_id, p, rel_label, answers = all_q_ids[i], all_paragraphs[i], all_rels[i], all_answers[i]
            text_ids = all_encodings["input_ids"][ei]
            type_ids = all_encodings["token_type_ids"][ei]
            mask_ids = all_encodings["attention_mask"][ei]
            offset_map = all_encodings["offset_mapping"][ei]
            if q_id not in collector:
                collector[q_id] = {"rel": [], "irrel": [], "unk": []}
            if rel_label == 1:
                for a in answers:
                    start, a_text = a["start"], a["text"]
                    end = start + len(a_text) - 1
                    start_label = ((1 - type_ids) * mask_ids).sum().item()
                    end_label = len(offset_map) - ((1 - type_ids) * (1 - mask_ids)).sum().item() - 2
                    if not(offset_map[start_label][0] <= start and offset_map[end_label][1] > end):
                        rel = 0
                    else:
                        while start_label < len(offset_map) and offset_map[start_label][0] <= start:
                            start_label += 1
                        start_label -= 1
                        while offset_map[end_label][1] - 1 >= end:
                            end_label -= 1
                        end_label += 1
                        collector[q_id]["rel"].append({"q_id": q_id, "paragraph": p,    \
                                                "text_ids": text_ids, "type_ids": type_ids, "mask_ids": mask_ids,   \
                                                "offset_map": offset_map, "rel_label": rel_label, "answer": a_text, \
                                                "start_label": start_label, "end_label": end_label})
                        if a_text != p[offset_map[start_label][0]: offset_map[end_label][1]]:
                            print(q_id, a["text"], p[offset_map[start_label][0]: offset_map[end_label][1]], offset_map[start_label], offset_map[end_label], flush=True)
            if rel_label == 0:
                collector[q_id]["irrel"].append({"q_id": q_id, "paragraph": p,    \
                                                "text_ids": text_ids, "type_ids": type_ids, "mask_ids": mask_ids,   \
                                                "offset_map": offset_map, "rel_label": rel_label, "answer": '', \
                                                "start_label": args.max_seq_len, "end_label": args.max_seq_len})
            if rel_label == -1:
                collector[q_id]["unk"].append({"q_id": q_id, "paragraph": p,    \
                                                "text_ids": text_ids, "type_ids": type_ids, "mask_ids": mask_ids,   \
                                                "offset_map": offset_map, "rel_label": rel_label, "answer": '', \
                                                "start_label": args.max_seq_len, "end_label": args.max_seq_len})

        #self.data = self.sample_data()
    
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
                                #if a["text"] != p[offset_map[start_label][0]: offset_map[end_label][1]]:
                                print(a["text"], p[offset_map[start_label][0]: offset_map[end_label][1]], flush=True)
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
            print(bb - aa, cc - bb, flush=True)
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


