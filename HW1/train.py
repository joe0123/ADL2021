import os
import sys
import argparse
import logging
import random
import numpy as np
import json
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_utils import Vocab
from dataset import IntentDataset
from model import IntentGRU

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

    initializers = {
        "xavier_uniform_": torch.nn.init.xavier_uniform_,
        "xavier_normal_": torch.nn.init.xavier_normal,
        "orthogonal_": torch.nn.init.orthogonal_,
    }

    criterions = {
        "intent": nn.CrossEntropyLoss(),
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
    
    args.dataset = datasets[args.task]
    args.model_class = models[args.task]
    args.initializer = initializers[args.initializer]
    args.criterion = criterions[args.task]
    args.optimizer = optimizers[args.optimizer]
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device is None else torch.device(args.device)
    
    return args


class Trainer:
    def __init__(self, args):
        self.args = args
        vocab = pickle.load(open(os.path.join(args.cache_dir, "vocab.pkl"), "rb"))
        intent_idx = json.load(open(os.path.join(args.cache_dir, "intent2idx.json"), 'r'))
        embed_matrix = torch.load(os.path.join(args.cache_dir, "embeddings.pt"))
        
        self.train_dataset = args.dataset(args, "train", vocab, intent_idx)
        self.eval_dataset = args.dataset(args, "eval", vocab, intent_idx)
        self.model = args.model_class(embed_matrix, self.train_dataset.num_classes, args).to(args.device)
        
        self.best_ckpt = os.path.join(args.ckpt_dir, "best.ckpt")
        logger.info('\n')
    
    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.args.initializer(p)
                else:
                    stdv = 1. / np.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)
    
    def train(self, criterion, optimizer, train_dataloader, eval_dataloader):
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
                outputs = self.model(inputs, input_lens)
                targets = targets.to(self.args.device)
                 
                optimizer.zero_grad()
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_n += len(outputs)
                train_loss += loss.item() * len(outputs)
                if global_step % self.args.log_step == 0 or i == len(train_dataloader):
                    train_loss = train_loss / total_n
                    logger.info("Train | Loss: {:.5f}".format(train_loss))

            eval_acc = self.eval(eval_dataloader)
            logger.info("Valid | Acc: {:.5f}".format(eval_acc))
            if eval_acc > max_eval_acc:
                max_eval_acc = eval_acc
                torch.save(self.model.state_dict(), self.best_ckpt)
                logger.info("Saving model to {}...".format(self.best_ckpt))
            global_step = 0

    def eval(self, dataloader):
        all_targets, all_outputs = None, None
        self.model.eval()
        with torch.no_grad():
            for i, (_, inputs, input_lens, targets) in enumerate(dataloader):
                inputs = inputs.to(self.args.device)
                input_lens = input_lens.to(self.args.device)
                outputs = self.model(inputs, input_lens)
                targets = targets.to(self.args.device)
                
                if all_targets is None:
                    all_targets = targets
                    all_outputs = outputs
                else:
                    all_targets = torch.cat((all_targets, targets), dim=0)
                    all_outputs = torch.cat((all_outputs, outputs), dim=0)

        all_targets = all_targets.cpu()
        all_outputs = torch.argmax(all_outputs, dim=-1).cpu()
        acc = (all_outputs == all_targets).float().mean()
        
        return acc

    def run(self):
        train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size, \
                                collate_fn=self.train_dataset.collate_fn, shuffle=True, num_workers=8)
        eval_dataloader = DataLoader(dataset=self.eval_dataset, batch_size=self.args.batch_size, \
                                collate_fn=self.eval_dataset.collate_fn, shuffle=False, num_workers=8)
        
        criterion = self.args.criterion
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.args.optimizer(params, lr=self.args.lr)#, weight_decay=self.args.l2reg)
        
        self._reset_params()
        logger.info('>' * 100)
        logger.info("Start training...")
        self.train(criterion, optimizer, train_dataloader, eval_dataloader)
 
if __name__ == "__main__":
# Read hyperparameters from CMD
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["intent", "slot"], required=True, type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--cache_dir", default="./cache", type=str)
    parser.add_argument("--ckpt_dir", default="./ckpt", type=str)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--initializer", default="orthogonal_", type=str)
    parser.add_argument("--epoch_num", default=100, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--l2reg", default=0, type=float, help="0 is recommended, or the training might fail")
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
