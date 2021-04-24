import os
import sys
import argparse
import logging
import random
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast#, BertForQuestionAnswering
from time import strftime, localtime

from dataset import build_datasets
from model import QABert
from optims import *
import evaluate as ev

def seed_config(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dir_mapping(args):
    args.ckpt_dir = os.path.join(args.ckpt_dir, strftime("%m%d-%H%M", localtime()))
    os.makedirs(args.ckpt_dir, exist_ok=True)

    return args

def class_mapping(args):
    bert_ckpt_names = {
        "bert_base": "bert-base-chinese",
    }

    bert_tokenizers = {
        "bert_base": BertTokenizerFast,
    }

    bert_models = {
        "bert_base": QABert,
        #"bert_base": BertForQuestionAnswering,
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

    schedulers = {
        "const": get_constant_schedule_with_warmup,
        "linear": get_linear_schedule_with_warmup,
        "cosine": get_cosine_schedule_with_warmup,
        "invexp": get_invexp_schedule_with_warmup,
    }
    
    bert_ckpt_name = bert_ckpt_names[args.pretrained_bert]
    args.bert_tokenizer = bert_tokenizers[args.pretrained_bert].from_pretrained(bert_ckpt_name, do_lower_case=True)
    args.bert_model = bert_models[args.pretrained_bert].from_pretrained(bert_ckpt_name)
    args.initializer = initializers[args.init_name]
    args.optimizer = optimizers[args.opt_name]
    args.scheduler = schedulers[args.sched_name]
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device is None else torch.device(args.device)

    return args


class Trainer:
    def __init__(self, args):
        self.args = args
        if args.eval_ratio > 0:
            train_dataset, eval_dataset = build_datasets(args, "train")
            self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, \
                                collate_fn=train_dataset.collate_fn, shuffle=True, num_workers=8)
            self.eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=args.batch_size, \
                                collate_fn=eval_dataset.collate_fn, shuffle=False, num_workers=8)
            self.eval_tokenizer = ev.Tokenizer()
        else:
            train_dataset = build_datasets(args, "train")
            self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, \
                                collate_fn=train_dataset.collate_fn, shuffle=True, num_workers=8)
        
        self.model = args.bert_model
        self.model.to(args.device)
        self.best_ckpt = os.path.join(args.ckpt_dir, "best.ckpt")
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.args.optimizer(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.l2reg) 
        #TODO add scheduler
    
    def train(self):
        max_valid_acc = 0
        global_step = 0
        for epoch in range(self.args.epoch_num):
            logger.info('>' * 100)
            logger.info("Epoch {:02d} / {:02d}".format(epoch + 1, self.args.epoch_num))
            total_n, train_loss = 0, 0
            
            self.optimizer.zero_grad()
            for i, samples in enumerate(self.train_dataloader):
                self.model.train()
                global_step += 1
                
                text_ids = samples["text_ids"].to(args.device)
                type_ids = samples["type_ids"].to(args.device)
                mask_ids = samples["mask_ids"].to(args.device)
                start_labels = samples["start_labels"].to(args.device)
                end_labels = samples["end_labels"].to(args.device)
                outputs = self.model(text_ids, token_type_ids=type_ids, attention_mask=mask_ids, \
                                    start_positions=start_labels, end_positions=end_labels)
                loss = outputs["loss"]
                loss.backward()
                
                if global_step % self.args.update_step == 0 or i == len(self.train_dataloader) - 1:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                total_n += len(outputs)
                train_loss += loss.item()
                if global_step % self.args.log_step == 0 or i == len(self.train_dataloader) - 1:
                    train_loss = train_loss / total_n
                    logger.info("Train | Loss: {:.5f}".format(train_loss))
                
                # TODO Write eval
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
        logger.info('>' * 100)
        logger.info("Start training...")
        self.train()


if __name__ == "__main__":
# Read hyperparameters from CMD
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_data", default="./data/context.json", type=str)
    parser.add_argument("--train_data", default="./data/train.json", type=str)
    parser.add_argument("--ckpt_dir", default="./ckpt", type=str)
    parser.add_argument("--eval_ratio", default=0.2, type=float)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--epoch_num", default=5, type=int)
    parser.add_argument("--lr", default=3e-5, type=float, help="*e-5 are recommended")
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--l2reg", default=0.01, type=float)
    parser.add_argument("--opt_name", default="adam", type=str)
    parser.add_argument("--init_name", default="xavier_uniform_", type=str)
    parser.add_argument("--sched_name", default="linear", type=str)
    parser.add_argument("--update_step", default=32, type=int, help="number of steps to accum gradients before update")
    parser.add_argument("--log_step", default=500, type=int, help="number of steps to print the loss during training")
    parser.add_argument("--eval_step", default=1000, type=int, help="number of steps to evaluate the model during training")
    parser.add_argument("--warmup_ratio", default=0.1, type=float, help="ratio between 0 and 1 for warmup scheduling")
    parser.add_argument("--bert_dim", default=768, type=int)
    parser.add_argument("--pretrained_bert", default="bert_base", choices=["bert_base"], type=str)
    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument("--device", default="cuda:0", type=str, help="e.g. cuda:0")
    parser.add_argument("--seed", default=14, type=int, help="seed for reproducibility")
    args = parser.parse_args()

# Configure some settings
    seed_config(args)
    args = dir_mapping(args)
    args_file = os.path.join(args.ckpt_dir, "args.json")
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=4)
    args = class_mapping(args)

# Set logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    log_file = os.path.join(args.ckpt_dir, "log")
    logger.addHandler(logging.FileHandler(log_file))

# Start training
    trainer = Trainer(args)
    trainer.run()
