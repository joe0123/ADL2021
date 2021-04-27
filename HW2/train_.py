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
from transformers import BertTokenizerFast
from time import strftime, localtime

from dataset_ import build_datasets
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
    model_ckpt_names = {
        "bert_base": "bert-base-chinese",
    }

    tokenizers = {
        "bert_base": BertTokenizerFast,
    }

    models = {
        "bert_base": QABert,
        #"bert_base": BertForQuestionAnswering,
    }

    schedulers = {
        "const": get_constant_schedule_with_warmup,
        "linear": get_linear_schedule_with_warmup,
        "cosine": get_cosine_schedule_with_warmup,
        "invexp": get_invexp_schedule_with_warmup,
    }
    
    model_ckpt_name = model_ckpt_names[args.pretrained_model]
    args.tokenizer = tokenizers[args.pretrained_model].from_pretrained(model_ckpt_name, do_lower_case=True)
    args.model = models[args.pretrained_model].from_pretrained(model_ckpt_name)
    args.scheduler = schedulers[args.sched_name]
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device is None else torch.device(args.device)

    return args


class Trainer:
    def __init__(self, args):
        self.args = args
        if args.valid_ratio > 0:
            train_dataset, valid_dataset = build_datasets(args, "train")
            self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, \
                                                collate_fn=train_dataset.collate_fn, shuffle=True)
            self.valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.valid_batch_size, \
                                                collate_fn=valid_dataset.collate_fn, shuffle=False)
            self.eval_tokenizer = ev.Tokenizer()
        else:
            train_dataset = build_datasets(args, "train")
            self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, \
                                                collate_fn=train_dataset.collate_fn, shuffle=True)
        
        self.model = args.model
        self.model.to(args.device)
        self.best_ckpt = os.path.join(args.ckpt_dir, "best.ckpt")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, \
                                            eps=1e-6, weight_decay=self.args.l2reg)
        total_update_steps = np.ceil(len(self.train_dataloader) / args.update_step) * args.epoch_num
        self.scheduler = args.scheduler(self.optimizer, total_update_steps * args.warmup_ratio, total_update_steps)
    
    def train(self):
        max_valid_em = 0
        for epoch in range(self.args.epoch_num):
            logger.info('>' * 100)
            logger.info("Epoch {:02d} / {:02d}".format(epoch + 1, self.args.epoch_num))
            
            total_rel_loss, total_qa_loss, total_loss = 0, 0, 0
            self.optimizer.zero_grad()
            for step, samples in enumerate(self.train_dataloader, 1):
                self.model.train()
                text_ids = samples["text_ids"].to(self.args.device)
                type_ids = samples["type_ids"].to(self.args.device)
                mask_ids = samples["mask_ids"].to(self.args.device)
                rel_labels = samples["rel_labels"].to(self.args.device)
                start_labels = samples["start_labels"].to(self.args.device)
                end_labels = samples["end_labels"].to(self.args.device)
                outputs = self.model(text_ids, token_type_ids=type_ids, attention_mask=mask_ids, \
                                    rel_labels=rel_labels, start_labels=start_labels, end_labels=end_labels)
                
                rel_loss = outputs["rel_loss"]
                qa_loss = (outputs["start_loss"] + outputs["end_loss"]) / 2
                loss = (rel_loss + qa_loss) / 2
                total_rel_loss += rel_loss.item()
                total_qa_loss += qa_loss.item()
                total_loss += loss.item()

                if len(self.train_dataloader) - step + 1 < self.args.update_step:
                    loss = loss / (len(self.train_dataloader) % self.args.update_step)
                else:
                    loss = loss / self.args.update_step
                loss.backward()
                
                if step % self.args.update_step == 0 or step == len(self.train_dataloader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                
                if step % self.args.log_step == 0 or step == len(self.train_dataloader):
                    total_loss /= step
                    total_rel_loss /= step
                    total_qa_loss /= step
                    logger.info("Train | Loss: {:.5f} (Rel: {:.5f}, QA: {:.5f})".format(total_loss, 
                                                                                        total_rel_loss,
                                                                                        total_qa_loss))
                if self.args.valid_ratio > 0 and step % self.args.eval_step == 0 or step == len(self.train_dataloader):
                    valid_scores = self.eval(self.valid_dataloader)
                    valid_em, valid_f1 = valid_scores["em"], valid_scores["f1"]
                    logger.info("Valid | EM: {:.5f}, F1: {:.5f}".format(valid_em, valid_f1))
                    if valid_em > max_valid_em:
                        max_valid_em = valid_em
                        torch.save(self.model.state_dict(), self.best_ckpt)
                        logger.info("Saving model to {}...".format(self.best_ckpt))

    def eval(self, dataloader):
        all_targets, all_outputs = dict(), dict()
        self.model.eval()
        with torch.no_grad():
            for samples in dataloader:
                text_ids = samples["text_ids"].to(args.device)
                type_ids = samples["type_ids"].to(args.device)
                mask_ids = samples["mask_ids"].to(args.device)
                rel_labels = samples["rel_labels"].to(args.device)
                start_labels = samples["start_labels"].to(args.device)
                end_labels = samples["end_labels"].to(args.device)
                outputs = self.model(text_ids, token_type_ids=type_ids, attention_mask=mask_ids, \
                                    rel_labels=rel_labels, start_labels=start_labels, end_labels=end_labels)
                
                rels = outputs["rel_logits"].cpu().tolist()
                starts = (torch.argmax(outputs["start_logits"], dim=-1)).cpu().tolist()
                ends = (torch.argmax(outputs["end_logits"], dim=-1)).cpu().tolist()
                #rels = rel_labels.cpu().tolist()
                #starts = start_labels.cpu().tolist()
                #ends = end_labels.cpu().tolist()
                # TODO ensure start - end < 30
                for q_id, p, offset_map, ans, rel, start, end \
                        in zip(samples["q_ids"], samples["paragraphs"], samples["offset_maps"], samples["answers"], \
                                rels, starts, ends):
                    if q_id not in all_targets:
                        all_targets[q_id] = {"answers": [ans] if len(ans) > 0 else []}
                        all_outputs[q_id] = (rel, p[offset_map[start][0]: offset_map[end][1]])
                    else:
                        if len(ans) > 0:
                            all_targets[q_id]["answers"].append(ans)
                        if rel > all_outputs[q_id][0]:
                            all_outputs[q_id] = (rel, p[offset_map[start][0]: offset_map[end][1]])
        
        for k in all_outputs:
            all_outputs[k] = all_outputs[k][1]
        scores = ev.compute_metrics(all_targets, all_outputs, self.eval_tokenizer)
                        
        return scores

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
    parser.add_argument("--valid_ratio", default=0.2, type=float)
    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument("--valid_batch_size", default=48, type=int)
    parser.add_argument("--epoch_num", default=10, type=int)
    parser.add_argument("--lr", default=5e-5, type=float, help="*e-5 are recommended")
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--l2reg", default=0.01, type=float)
    parser.add_argument("--sched_name", default="linear", type=str)
    parser.add_argument("--update_step", default=32, type=int, help="number of steps to accum gradients before update")
    parser.add_argument("--log_step", default=1000, type=int, help="number of steps to print the loss during training")
    parser.add_argument("--eval_step", default=1, type=int, help="number of steps to evaluate the model during training")
    #parser.add_argument("--eval_step", default=6000, type=int, help="number of steps to evaluate the model during training")
    parser.add_argument("--warmup_ratio", default=0.1, type=float, help="ratio between 0 and 1 for warmup scheduling")
    parser.add_argument("--irrel_ratio", default=5, type=float, help="num of irrel: num of rel")
    parser.add_argument("--pretrained_model", default="bert_base", choices=["bert_base"], type=str)
    parser.add_argument("--max_seq_len", default=384, type=int)
    parser.add_argument("--stride", default=128, type=int)
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
