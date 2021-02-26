import os
import sys
import argparse
import logging
import random
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from transformers import get_linear_schedule_with_warmup

from data_utils import Tokenizer4Bert, SentimentDataset
from model import BERT_based
from loss_utils import smooth

def seed_config(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    initializers = {
        "xavier_uniform_": torch.nn.init.xavier_uniform_,
        "xavier_normal_": torch.nn.init.xavier_normal,
        "orthogonal_": torch.nn.init.orthogonal_,
    }

    optimizers = {
        "adadelta": torch.optim.Adadelta,  # default lr=1.0
        "adagrad": torch.optim.Adagrad,  # default lr=0.01
        "adam": torch.optim.Adam,  # default lr=0.001
        "adamax": torch.optim.Adamax,  # default lr=0.002
        "asgd": torch.optim.ASGD,  # default lr=0.01
        "rmsprop": torch.optim.RMSprop,  # default lr=0.01
        "sgd": torch.optim.SGD,
    }

    warmup_scheduler = {
        "linear": get_linear_schedule_with_warmup,
    }

    args.input_cols = ["bert_text_ids", "bert_mask_ids"]
    args.dataset = SentimentDataset
    args.class_num = 1
    args.model_class = BERT_based
    bert_ckpt = bert_ckpts[args.pretrained_bert]
    args.bert_tokenizer = BertTokenizer.from_pretrained(bert_ckpt)
    args.bert_model = bert_models[args.pretrained_bert].from_pretrained(bert_ckpt)
    args.initializer = initializers[args.initializer]
    args.criterion = nn.BCELoss()
    args.optimizer = optimizers[args.optimizer]
    args.warmup_scheduler = warmup_scheduler[args.warmup]
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device is None else torch.device(args.device)
    
    return args


class Trainer:
    def __init__(self, args):
        self.args = args
        tokenizer = Tokenizer4Bert(args.max_seq_len, args.bert_tokenizer)
        bert = args.bert_model
        self.bert_type = type(bert)
        self.model = args.model_class(bert, args).to(args.device)
        
        self.train_dataset = args.dataset(args, "train", tokenizer)
        self.valid_dataset = args.dataset(args, "valid", tokenizer)
        
        self.best_ckpt = os.path.join(args.ckpt_dir, "best.ckpt")
        logger.info('\n')
    
    def _reset_params(self):
        for child in self.model.children():
            if type(child) != self.bert_type:
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.args.initializer(p)
                        else:
                            stdv = 1. / np.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)
    
    def train(self, criterion, optimizer, warmup_scheduler, train_dataloader, valid_dataloader):
        max_valid_acc = 0
        global_step = 0
        for epoch in range(self.args.epoch_num):
            logger.info('>' * 100)
            logger.info("Epoch {:02d} / {:02d}".format(epoch + 1, self.args.epoch_num))
            correct_n, total_n, train_loss = 0, 0, 0
            
            optimizer.zero_grad()
            for i, sample in enumerate(train_dataloader):
                self.model.train()
                global_step += 1

                inputs = [sample[col].to(self.args.device) for col in self.args.input_cols]
                targets = sample["polarity"].to(self.args.device)
                outputs = self.model(inputs)
                
                if args.smooth_ratio > 0:
                    loss = criterion(outputs.float(), smooth(targets, args.smooth_ratio).float())
                else:
                    loss = criterion(outputs.float(), targets.float())
                loss.backward()
                
                if global_step % self.args.update_step == 0 or i == len(train_dataloader) - 1:
                    optimizer.step()
                    warmup_scheduler.step()
                    optimizer.zero_grad()
                
                total_n += len(outputs)
                train_loss += loss.item() * len(outputs)
                if global_step % self.args.log_step == 0 or i == len(train_dataloader) - 1:
                    train_loss = train_loss / total_n
                    logger.info("Train | Loss: {:.5f}".format(train_loss))

                if global_step % self.args.eval_step == 0 or i == len(train_dataloader) - 1:
                    valid_acc = self.eval(valid_dataloader)
                    logger.info("Valid | Acc: {:.5f}".format(valid_acc))
                    if valid_acc > max_valid_acc:
                        max_valid_acc = valid_acc
                        torch.save(self.model.state_dict(), self.best_ckpt)
                        logger.info("Saving model to {}...".format(self.best_ckpt))
            global_step = 0

    def eval(self, dataloader):
        all_targets, all_outputs = None, None
        self.model.eval()
        with torch.no_grad():
            for i, sample in enumerate(dataloader):
                inputs = [sample[col].to(self.args.device) for col in self.args.input_cols]
                targets = sample["polarity"].to(self.args.device)
                outputs = self.model(inputs)

                if all_targets is None:
                    all_targets = targets
                    all_outputs = outputs
                else:
                    all_targets = torch.cat((all_targets, targets), dim=0)
                    all_outputs = torch.cat((all_outputs, outputs), dim=0)

        all_targets = all_targets.cpu()
        all_outputs = torch.round(all_outputs).cpu()
        acc = (all_outputs == all_targets).float().mean()
        return acc

    def run(self):
        train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size, \
                                shuffle=True, num_workers=8)
        valid_dataloader = DataLoader(dataset=self.valid_dataset, batch_size=self.args.batch_size, \
                                shuffle=False, num_workers=8)
        
        criterion = self.args.criterion
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.args.optimizer(params, lr=self.args.lr, weight_decay=self.args.l2reg)
        total_step = np.ceil(len(train_dataloader) / self.args.update_step) * self.args.epoch_num
        warmup_scheduler = self.args.warmup_scheduler(optimizer, \
                            num_warmup_steps=int(self.args.warmup_ratio * total_step), num_training_steps=total_step)

        self._reset_params()
        logger.info('>' * 100)
        logger.info("Start training...")
        self.train(criterion, optimizer, warmup_scheduler, train_dataloader, valid_dataloader)
       
        
if __name__ == "__main__":
# Read hyperparameters from CMD
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", default="../data/train.csv", type=str)
    parser.add_argument("--valid_data", default="../data/dev.csv", type=str)
    parser.add_argument("--ckpt_dir", default="./ckpt", type=str)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--warmup", default="linear", type=str)
    parser.add_argument("--initializer", default="xavier_uniform_", type=str)
    parser.add_argument("--epoch_num", default=5, type=int)
    parser.add_argument("--batch_size", default=16, type=int, help="try 16, 32, 64")
    parser.add_argument("--lr", default=3e-5, type=float, help="*e-5 are recommended")
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--l2reg", default=0.01, type=float)
    parser.add_argument("--update_step", default=8, type=int, help="number of steps to accum gradients before update")
    parser.add_argument("--log_step", default=500, type=int, help="number of steps to print the loss during training")
    parser.add_argument("--eval_step", default=1000, type=int, help="number of steps to evaluate the model during training")
    parser.add_argument("--bert_dim", default=768, type=int)
    parser.add_argument("--pretrained_bert", default="bert_base", \
                        choices=["bert_base", "bert_ml_base", "roberta_base"], type=str)
    parser.add_argument("--max_seq_len", default=384, type=int)
    parser.add_argument("--device", default="cuda:0", type=str, help="e.g. cuda:0")
    parser.add_argument("--seed", default=14, type=int, help="seed for reproducibility")
    parser.add_argument("--warmup_ratio", default=0.1, type=float, help="ratio between 0 and 1 for warmup scheduling")
    parser.add_argument("--smooth_ratio", default=0, type=float, help="ratio between 0 and 1 for LSR")
    args = parser.parse_args()

# Configure some settings
    seed_config(args)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    args_file = os.path.join(args.ckpt_dir, "args.json")
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=4)
    args = args_mapping(args)

# Set logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    log_file = os.path.join(args.ckpt_dir, "log")
    logger.addHandler(logging.FileHandler(log_file))

# Start training
    trainer = Trainer(args)
    trainer.run()

