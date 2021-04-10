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

from data_utils import Vocab
from dataset import IntentDataset, SlotDataset
from model import IntentGRU, SlotGRU
from torchcrf import CRF
from optim import *

def seed_config(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dir_mapping(args):
    args.data_dir = os.path.join(args.data_dir, args.task)
    args.cache_dir = os.path.join(args.cache_dir, args.task)
    args.ckpt_dir = os.path.join(args.ckpt_dir, args.task, strftime("%m%d-%H%M", localtime()))
    os.makedirs(args.ckpt_dir, exist_ok=True)

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

    initializers = {
        "xavier_uniform_": torch.nn.init.xavier_uniform_,
        "xavier_normal_": torch.nn.init.xavier_normal,
        "orthogonal_": torch.nn.init.orthogonal_,
    }

    criterions = {
        "ce": nn.CrossEntropyLoss,
        "crf": CRF,
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
    
    args.dataset = datasets[args.task]
    args.model_class = models[args.task]
    args.initializer = initializers[args.init_name]
    args.criterion = criterions[args.cri_name]
    args.optimizer = optimizers[args.opt_name]
    args.embed_optimizer = optimizers[args.embed_opt_name]
    args.scheduler = schedulers[args.sched_name]
    args.embed_scheduler = schedulers[args.embed_sched_name]
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device is None else torch.device(args.device)
    
    return args


class Trainer:
    def __init__(self, args):
        self.args = args
        
        if args.no_eval:
            self.train_dataset = args.dataset(args, ["train", "eval"], label_type="tensor")
        else:
            self.train_dataset = args.dataset(args, ["train"], label_type="tensor")
            self.eval_dataset = args.dataset(args, ["eval"], label_type="list")
        self.model = args.model_class(self.train_dataset.num_classes, self.train_dataset.label2id.get("[PAD]", None) , args)
        self.model.to(args.device)
        
        self.best_ckpt = os.path.join(args.ckpt_dir, "best.ckpt")
    
    def _reset_params(self):
        for n, p in self.model.named_parameters():
            if ("criterion" not in n) and ("embed" not in n) and p.requires_grad:
                if len(p.shape) > 1:
                    self.args.initializer(p)
                else:
                    stdv = 1. / np.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)
    
    def train(self, optimizers, schedulers, train_dataloader, eval_dataloader):
        max_eval_acc = 0
        global_step = 0
        for epoch in range(self.args.epoch_num):
            logger.info('>' * 100)
            logger.info("Epoch {:03d} / {:03d}".format(epoch + 1, self.args.epoch_num))
            correct_n, total_n, train_loss = 0, 0, 0
            
            for i, (_, inputs, input_lens, targets) in enumerate(train_dataloader):
                self.model.train()
                global_step += 1
                inputs = inputs.to(self.args.device)
                input_lens = input_lens.to(self.args.device)
                targets = targets.to(self.args.device)
                
                for optimizer in optimizers:
                    optimizer.zero_grad()
                loss = self.model.compute_loss(inputs, input_lens, targets)
                loss.backward()
                for optimizer in optimizers:
                    optimizer.step()

                total_n += targets.shape[0]
                train_loss += loss.item() * targets.shape[0]
                if global_step % self.args.log_step == 0 or i == len(train_dataloader):
                    train_loss = train_loss / total_n
                    logger.info("Train | Loss: {:.5f}".format(train_loss))
            if eval_dataloader:
                eval_acc = self.eval(eval_dataloader)
                logger.info("Valid | Acc: {:.5f}".format(eval_acc))
                if eval_acc >= max_eval_acc:
                    max_eval_acc = eval_acc
                    torch.save(self.model.state_dict(), self.best_ckpt)
                    logger.info("Saving model to {}...".format(self.best_ckpt))
            else:
                torch.save(self.model.state_dict(), self.best_ckpt)
                logger.info("Saving model to {}...".format(self.best_ckpt))
            global_step = 0
            for scheduler in schedulers:
                scheduler.step()

    def eval(self, dataloader):
        all_targets, all_outputs = None, None
        self.model.eval()
        acc = 0
        with torch.no_grad():
            for i, (_, inputs, input_lens, targets) in enumerate(dataloader):
                inputs = inputs.to(self.args.device)
                input_lens = input_lens.to(self.args.device)
                acc += self.model.score(inputs, input_lens, targets)
        acc /= len(dataloader.dataset)        
        return acc

    def run(self):
        train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size, \
                                collate_fn=self.train_dataset.collate_fn, shuffle=True, num_workers=8)
        if not self.args.no_eval:
            eval_dataloader = DataLoader(dataset=self.eval_dataset, batch_size=self.args.batch_size, \
                                collate_fn=self.eval_dataset.collate_fn, shuffle=False, num_workers=8)
        
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
    parser.add_argument("--task", choices=["intent", "slot"], required=True, type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--cache_dir", default="./cache", type=str)
    parser.add_argument("--ckpt_dir", default="./ckpt", type=str)
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--cri_name", default="ce", type=str)
    parser.add_argument("--opt_name", default="adam", type=str)
    parser.add_argument("--embed_opt_name", default="adam", type=str)
    parser.add_argument("--sched_name", default="const", type=str)
    parser.add_argument("--embed_sched_name", default="cosine", type=str)
    parser.add_argument("--init_name", default="orthogonal_", type=str)
    parser.add_argument("--epoch_num", default=100, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--embed_lr", default=2e-5, type=float)
    parser.add_argument("--warmup_epoch", default=0, type=int)
    parser.add_argument("--embed_warmup_epoch", default=50, type=int)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--l2reg", default=1e-5, type=float)
    parser.add_argument("--log_step", default=50, type=int, help="number of steps to print the loss during training")
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--max_seq_len", type=int)
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
