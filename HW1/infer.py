import os
import sys
import argparse
import logging
import random
import numpy as np
import json
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from data_utils import Vocab
from dataset import IntentDataset
from model import IntentGRU

def args_loading(test_args, ckpt_dir):
    args = argparse.Namespace()
    with open(os.path.join(ckpt_dir, "args.json"), 'r') as f:
        train_args = json.load(f)

    for k, v in train_args.items():
        vars(args)[k] = v

    for k, v in vars(test_args).items():
        vars(args)[k] = v

    return args

def dir_mapping(args):
    args.ckpt_dir = os.path.join(args.ckpt_dir, args.task)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    return args

def class_mapping(args):
    datasets = {
        "intent": IntentDataset,
    }
    
    models = {
        "intent": IntentGRU,
    }
        
    args.dataset = datasets[args.task]
    args.model_class = models[args.task]
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device is None else torch.device(args.device)
    
    return args

class Tester:
    def __init__(self, args):
        self.args = args
        vocab = pickle.load(open(os.path.join(args.cache_dir, "vocab.pkl"), "rb"))
        intent_label = json.load(open(os.path.join(args.cache_dir, "intent2idx.json"), 'r'))
        embed_matrix = torch.load(os.path.join(args.cache_dir, "embeddings.pt"))
        
        self.test_dataset = args.dataset(args, "test", vocab, intent_label)
        self.model = args.model_class(embed_matrix, self.test_dataset.num_classes, args).to(args.device)
        
        best_ckpt = os.path.join(args.ckpt_dir, "best.ckpt")
        self.model.load_state_dict(torch.load(best_ckpt))
        logger.info('\n')

    def infer(self, dataloader):
        all_ids, all_outputs = None, None
        self.model.eval()
        with torch.no_grad():
            for i, (ids, inputs, input_lens, _) in tqdm(enumerate(dataloader)):
                inputs = inputs.to(self.args.device)
                input_lens = input_lens.to(self.args.device)
                outputs = self.model(inputs, input_lens)
                if all_outputs is None:
                    all_ids = ids
                    all_outputs = outputs
                else:
                    all_ids += ids
                    all_outputs = torch.cat((all_outputs, outputs), dim=0)
        
        all_labels = torch.argmax(all_outputs, dim=-1).int().cpu().tolist()
        all_labels = [dataloader.dataset.label_intent[label] for label in all_labels]
        
        return dict(zip(all_ids, all_labels))
    
    def run(self):
        test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=self.args.batch_size, \
                                collate_fn=self.test_dataset.collate_fn, shuffle=False, num_workers=8)
        logger.info("Start testing...")
        result = self.infer(test_dataloader)
        
        return result

if __name__ == "__main__":
# Read hyperparameters from CMD
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["intent", "slot"], required=True, type=str)
    parser.add_argument("--ckpt_dir", default="./ckpt", type=str)
    parser.add_argument("--pred_file", default="./result.csv", type=str)
    parser.add_argument("--batch_size", default=256, type=int, help="test batch size")
    parser.add_argument("--device", default="cuda:0", type=str, help="e.g. cuda:0")
    test_args = parser.parse_args()
    
# Set logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))

# Start making inference
    test_args = dir_mapping(test_args)
    args = args_loading(test_args, test_args.ckpt_dir)    # Load args from the saved checkpoint
    args = class_mapping(args)
    tester = Tester(args)
    result = tester.run()

# Writing file
    with open(args.pred_file, 'w') as f:
        f.write("id,intent\n")
        for idx, intent in sorted(result.items()):
            f.write("{},{}\n".format(idx, intent))

