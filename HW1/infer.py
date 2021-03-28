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
from dataset import IntentDataset, SlotDataset
from model import IntentGRU, SlotGRU

def args_loading(test_args, ckpt_dir):
    args = argparse.Namespace()
    with open(os.path.join(ckpt_dir, "args.json"), 'r') as f:
        train_args = json.load(f)

    for k, v in train_args.items():
        vars(args)[k] = v

    for k, v in vars(test_args).items():
        vars(args)[k] = v

    return args


def class_mapping(args):
    datasets = {
        "intent": IntentDataset,
        "slot": SlotDataset
    }
    
    models = {
        "intent": IntentGRU,
        "slot": SlotGRU
    }
        
    args.dataset = datasets[args.task]
    args.model_class = models[args.task]
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device is None else torch.device(args.device)
    
    return args

class Tester:
    def __init__(self, args):
        self.args = args
        self.test_dataset = args.dataset(args, ["test"])
        self.model = args.model_class(self.test_dataset.num_classes, self.test_dataset.label2id.get("[PAD]", None) , args)
        self.model.to(args.device)
        
        best_ckpt = os.path.join(args.ckpt_dir, "best.ckpt")
        self.model.load_state_dict(torch.load(best_ckpt))

    def infer(self, dataloader):
        all_ids, all_preds = None, None
        self.model.eval()
        with torch.no_grad():
            for i, (ids, inputs, input_lens, _) in enumerate(dataloader):
                inputs = inputs.to(self.args.device)
                input_lens = input_lens.to(self.args.device)
                preds = self.model.predict(inputs, input_lens).cpu().tolist()
                if self.args.task == "slot":
                    preds = [pred[:input_len] for pred, input_len in zip(preds, input_lens.cpu().tolist())]
                if all_preds is None:
                    all_ids = ids
                    all_preds = preds
                else:
                    all_ids += ids
                    all_preds += preds
 
        if self.args.task == "intent":
            all_preds = [dataloader.dataset.id2label[label_id] for label_id in all_preds]
        elif self.args.task == "slot":
            all_preds = [' '.join([dataloader.dataset.id2label[label_id] for label_id in preds]) for preds in all_preds]
        
        return dict(zip(all_ids, all_preds))
    
    def run(self):
        test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=self.args.batch_size, \
                                collate_fn=self.test_dataset.collate_fn, shuffle=False, num_workers=8)
        logger.info("Start testing...")
        result = self.infer(test_dataloader)
        
        return result

if __name__ == "__main__":
# Read hyperparameters from CMD
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True, type=str)
    parser.add_argument("--pred_file", default="./result.csv", type=str)
    parser.add_argument("--batch_size", default=256, type=int, help="test batch size")
    parser.add_argument("--device", default="cuda:0", type=str, help="e.g. cuda:0")
    test_args = parser.parse_args()

# Set logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))

# Start making inference
    args = args_loading(test_args, test_args.ckpt_dir)    # Load args from the saved checkpoint
    args = class_mapping(args)
    tester = Tester(args)
    result = tester.run()

# Writing file
    with open(args.pred_file, 'w') as f:
        if args.task == "intent":
            f.write("id,intent\n")
        else:
            f.write("id,tags\n")
        for idx, label in sorted(result.items()):
            f.write("{},{}\n".format(idx, label))

