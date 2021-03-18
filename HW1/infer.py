import os
import sys
from tqdm import tqdm
import argparse
import logging
import random
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_utils import Dictionary, Tokenizer4LSTM, SentimentDataset
from model import LSTM


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
    args.dataset = SentimentDataset
    args.class_num = 1
    args.model_class = LSTM
    args.criterion = nn.BCELoss()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device is None else torch.device(args.device)
    
    return args

class Tester:
    def __init__(self, args):
        self.args = args
        dictionary = Dictionary(args.w2v_model)
        tokenizer = Tokenizer4LSTM(dictionary, args.max_seq_len)
        self.model = args.model_class(dictionary.embed_matrix, args).to(args.device)

        best_ckpt = os.path.join(args.ckpt_dir, "best.ckpt")
        self.model.load_state_dict(torch.load(best_ckpt))
        
        self.dataset = args.dataset(args, "test", tokenizer)
            
    def infer(self, dataloader):
        all_ids, all_outputs = None, None
        self.model.eval()
        with torch.no_grad():
            for i, sample in tqdm(enumerate(dataloader)):
                ids = sample["id"]
                inputs = sample["text_ids"].to(self.args.device)
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

