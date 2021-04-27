import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import time

def make_data_pool(args, context_data, ques_data, case):
    ss = time.time()
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
    print(time.time() - ss)
    
    ss = time.time()
    all_encodings = args.tokenizer(all_questions, all_paragraphs, return_tensors="pt", return_offsets_mapping=True, \
                        padding="max_length", truncation="only_second", max_length=args.max_seq_len,   \
                        stride=args.stride, return_overflowing_tokens=True)
    print(time.time() - ss)
    
    collector = dict()
    ss = time.time()
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
                    rel_label = 0
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
                    #if a_text != p[offset_map[start_label][0]: offset_map[end_label][1]]:
                    #    print(q_id, a["text"], p[offset_map[start_label][0]: offset_map[end_label][1]], offset_map[start_label], offset_map[end_label], flush=True)
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
    
    print(time.time() - ss)
    data_pool = [collector[k] for k in sorted(collector.keys())]

    return data_pool

def build_datasets(args, case):
    context_data = json.load(open(args.context_data, 'r'))
    ques_data = json.load(open(getattr(args, "{}_data".format(case)), 'r'))
    if case == "train" and args.valid_ratio > 0:
        ques_data = np.array(ques_data)
        ids = np.random.permutation(ques_data.shape[0])
        cut = int(ques_data.shape[0] * args.valid_ratio)
        train_ques_data = ques_data[ids[cut:]].tolist()
        train_data_pool = make_data_pool(args, context_data, train_ques_data, case)
        eval_ques_data = ques_data[ids[:cut]].tolist()
        eval_data_pool = make_data_pool(args, context_data, eval_ques_data, case)
        return QADataset(args, train_data_pool, case), QADataset(args, eval_data_pool, case)
    else:
        data_pool = make_data_pool(args, context_data, ques_data, case)
        return QADataset(args, data_pool, case)

class QADataset(Dataset):
    def __init__(self, args, data_pool, case):
        self.args = args
        self.case = case
        self.data_pool = data_pool
        self.sample_data()
    
    def sample_data(self):    #TODO implement with irrel_ratio ratio
        self.data = []
        for d in self.data_pool:
            if self.case == "train":
                for r in d["rel"]:
                    self.data.append(r)
                ir_ids = np.random.permutation(np.arange(len(d["irrel"]))).tolist()
                if self.args.irrel_ratio:
                    ir_size = min(len(d["rel"]) * self.args.irrel_ratio, len(ir_ids))
                else:
                    ir_size = len(ir_ids)
                for ir_id in ir_ids[:ir_size]:
                    self.data.append(d["irrel"][ir_id])
            else:
                for u in d["unk"]:
                    self.data.append(u)
    
    def collate_fn(self, samples):
        merged_samples = {"q_ids": [], "paragraphs": [], "text_ids": [], "type_ids": [], "mask_ids": [], "offset_maps": [], \
                        "rel_labels": [], "answers": [], "start_labels": [], "end_labels": []}
        for sample in samples:
            merged_samples["q_ids"].append(sample["q_id"])
            merged_samples["paragraphs"].append(sample["paragraph"])
            merged_samples["text_ids"].append(sample["text_ids"].unsqueeze(0))
            merged_samples["type_ids"].append(sample["type_ids"].unsqueeze(0))
            merged_samples["mask_ids"].append(sample["mask_ids"].unsqueeze(0))
            merged_samples["offset_maps"].append(sample["offset_map"].unsqueeze(0))
            merged_samples["rel_labels"].append(sample["rel_label"])
            merged_samples["answers"].append(sample["answer"])
            merged_samples["start_labels"].append(sample["start_label"])
            merged_samples["end_labels"].append(sample["end_label"])
        merged_samples["text_ids"] = torch.cat(merged_samples["text_ids"]).long()
        merged_samples["type_ids"] = torch.cat(merged_samples["type_ids"]).long()
        merged_samples["mask_ids"] = torch.cat(merged_samples["mask_ids"]).long()
        merged_samples["offset_maps"] = torch.cat(merged_samples["offset_maps"]).long()
        merged_samples["rel_labels"] = torch.FloatTensor(merged_samples["rel_labels"])
        merged_samples["start_labels"] = torch.LongTensor(merged_samples["start_labels"])
        merged_samples["end_labels"] = torch.LongTensor(merged_samples["end_labels"])

        return merged_samples
    
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


