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
from time import strftime, localtime

from dataset import QADataset
from optims import *

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
    
    args.dataset = QADataset
    args.model_class = None #TODO
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
            train_dataset = args.dataset(args, "train")
            self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, \
                                collate_fn=train_dataset.collate_fn, shuffle=True, num_workers=8)
        else:
            train_dataset, eval_dataset = args.dataset(args, "train")
            self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, \
                                collate_fn=train_dataset.collate_fn, shuffle=True, num_workers=8)
            self.eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=args.batch_size, \
                                collate_fn=eval_dataset.collate_fn, shuffle=False, num_workers=8)
        self.model = args.model_class(self.train_dataset.num_classes, self.train_dataset.label2id.get("[PAD]", None) , args)
        self.model.to(args.device)
        self.best_ckpt = os.path.join(args.ckpt_dir, "best.ckpt")
        # TODO mv other configuration from run to here
    
    def _reset_params(self):
        for n, p in self.model.named_parameters():
            if ("criterion" not in n) and ("embed" not in n) and p.requires_grad:
                if len(p.shape) > 1:
                    self.args.initializer(p)
                else:
                    stdv = 1. / np.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)
    
    def run(self):
        embed_params = [param for name, param in self.model.named_parameters() if "embed" in name]
        embed_optimizer = self.args.embed_optimizer(embed_params, lr=self.args.embed_lr, weight_decay=self.args.l2reg)
        other_params = [param for name, param in self.model.named_parameters() if "embed" not in name]
        other_optimizer = self.args.optimizer(other_params, lr=self.args.lr, weight_decay=self.args.l2reg)
        optimizers = [embed_optimizer, other_optimizer]

        embed_scheduler = self.args.embed_scheduler(embed_optimizer, \
                            num_warmup_steps=self.args.embed_warmup_epoch,  \
                            num_training_steps=self.args.epoch_num)
        other_scheduler = self.args.scheduler(other_optimizer, \
                            num_warmup_steps=self.args.warmup_epoch,  \
                            num_training_steps=self.args.epoch_num)
        schedulers = [embed_scheduler, other_scheduler]

        self._reset_params()
        logger.info('>' * 100)
        logger.info("Start training...")
        if self.args.no_eval:
            self.train(optimizers, schedulers, train_dataloader, None)
        else:
            self.train(optimizers, schedulers, train_dataloader, eval_dataloader)

if __name__ == "__main__":
# Read hyperparameters from CMD
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_data", default="./data/context.json", type=str)
    parser.add_argument("--train_data", default="./data/train.json", type=str)
    parser.add_argument("--ckpt_dir", default="./ckpt", type=str)
    parser.add_argument("--eval_ratio", default=0.2, type=float)
    parser.add_argument("--batch_size", default=16, type=int, help="try 16, 32, 64")
    parser.add_argument("--epoch_num", default=5, type=int)
    parser.add_argument("--lr", default=3e-5, type=float, help="*e-5 are recommended")
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--l2reg", default=0.01, type=float)
    parser.add_argument("--opt_name", default="adam", type=str)
    parser.add_argument("--init_name", default="xavier_uniform_", type=str)
    parser.add_argument("--sched_name", default="linear", type=str)
    parser.add_argument("--update_step", default=8, type=int, help="number of steps to accum gradients before update")
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
