import logging
import argparse
import json
import os
import sys
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

from data_utils import Tokenizer4Bert, SentimentDataset
from model import BERT_based
from loss_utils import smooth

def args_loading(test_args, ckpt_dir):
    args = argparse.Namespace()
    with open(os.path.join(ckpt_dir, "args.json"), 'r') as f:
        train_args = json.load(f)

    for k, v in train_args.items():
        vars(args)[k] = v

    for k, v in vars(test_args).items():
        vars(args)[k] = v

    return args

def args_mapping(args):
    bert_ckpts = {
        "bert_base": "bert-base-chinese",
        "bert_ml_base": "bert-base-multilingual-cased",
        "roberta_base": "hfl/chinese-roberta-wwm-ext",
    }

    bert_tokenizers = {
        "bert_base": BertTokenizer,
        "bert_ml_base": BertTokenizer,
        "roberta_base": BertTokenizer,
    }

    bert_models = {
        "bert_base": BertModel,
        "bert_ml_base": BertModel,
        "roberta_base": BertModel,
    }

    args.input_cols = ["bert_text_ids", "bert_mask_ids"]
    args.dataset = SentimentDataset
    args.class_num = 1
    args.model_class = BERT_based
    bert_ckpt = bert_ckpts[args.pretrained_bert]
    args.bert_tokenizer = BertTokenizer.from_pretrained(bert_ckpt)
    args.bert_model = bert_models[args.pretrained_bert].from_pretrained(bert_ckpt)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device is None else torch.device(args.device)

    return args

class Tester:
    def __init__(self, args):
        self.args = args

        tokenizer = Tokenizer4Bert(args.max_seq_len, args.bert_tokenizer)
        bert = args.bert_model
        self.model = args.model_class(bert, args).to(args.device)
        
        best_ckpt = os.path.join(args.ckpt_dir, "best.ckpt")
        self.model.load_state_dict(torch.load(best_ckpt))
        
        self.dataset = args.dataset(args, "test", tokenizer)
    
    def infer(self, dataloader):
        all_ids, all_outputs = None, None
        self.model.eval()
        with torch.no_grad():
            for i, sample in tqdm(enumerate(dataloader)):
                ids = sample["id"]
                inputs = [sample[col].to(self.args.device) for col in self.args.input_cols]
                outputs = self.model(inputs)
                if all_outputs is None:
                    all_ids = ids
                    all_outputs = outputs
                else:
                    all_ids += ids
                    all_outputs = torch.cat((all_outputs, outputs), dim=0)

        return dict(zip(all_ids, torch.round(all_outputs).int().cpu().tolist()))
    
    def run(self):
        dataloader = DataLoader(dataset=self.dataset, batch_size=self.args.batch_size, shuffle=False)
        logger.info("Start testing...")
        result = self.infer(dataloader)
        return result

if __name__ == "__main__":
# Read hyperparameters from CMD
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True, type=str)
    parser.add_argument("--test_data", default="../data/test.csv", type=str)
    parser.add_argument("--sample_data", default="../data/sample_submission.csv", type=str)
    parser.add_argument("--batch_size", default=256, type=int, help="test batch size")
    parser.add_argument("--device", default="cuda:0", type=str, help="e.g. cuda:0")
    test_args = parser.parse_args()
    
# Set logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))

# Start making inference
    args = args_loading(test_args, test_args.ckpt_dir)    # Load args from the saved checkpoint
    args = args_mapping(args)
    tester = Tester(args)
    result = tester.run()

# Writing file
    df = pd.read_csv(args.sample_data)
    df.Category = df.Id.map(result)
    df.to_csv("result.csv", index=False)

