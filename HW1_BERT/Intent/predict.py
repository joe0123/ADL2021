import os
import sys
import json
import argparse
import logging
import math
from functools import partial
from time import strftime, localtime
import datasets
from datasets import load_dataset, load_metric
from accelerate import Accelerator
import torch
from torch.utils.data.dataloader import DataLoader
import transformers
from transformers import (
    DataCollatorWithPadding,
    SchedulerType,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)

from data_utils import *

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--target_dir", type=str, required=True)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--out_file", type=str, default="./results.csv")
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_args()
    with open(os.path.join(args.target_dir, "args.json"), 'r') as f:
        train_args = json.load(f)
    for k, v in train_args.items():
        if not hasattr(args, k):
            vars(args)[k] = v

# Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
# Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    
# Setup logging, we only want one process per machine to log things on the screen.
# accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

# If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    
# Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(args.target_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.target_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.target_dir, config=config)

# Load and preprocess the dataset
    raw_datasets = load_dataset("json", data_files={"test": args.test_file})
    cols = raw_datasets["test"].column_names
    args.text_col, args.intent_col = "text", "intent"
    intent2id = config.label2id
    id2intent = config.id2label
    
    test_examples = raw_datasets["test"]
    #test_examples = test_examples.select(range(10))
    prepare_features = partial(prepare_features, args=args, tokenizer=tokenizer, intent2id=intent2id)
    test_dataset = test_examples.map(
        prepare_features,
        batched=True,
        num_proc=4,
        remove_columns=cols,
    )

# Create DataLoaders
    data_collator = DataCollatorWithPadding(tokenizer)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.test_batch_size)

# Prepare everything with our accelerator.
    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )

# Test!
    logger.info("\n******** Running predicting ********")
    logger.info(f"Num test examples = {len(test_dataset)}")
    
    test_dataset.set_format(columns=["attention_mask", "input_ids", "token_type_ids"])
    model.eval()
    all_predictions = []
    for step, data in enumerate(test_dataloader):
        with torch.no_grad():
            outputs = model(**data)
            predictions = outputs.logits.argmax(dim=-1)
            all_predictions += accelerator.gather(predictions).cpu().tolist()
    results = {example_id: id2intent[pred] for example_id, pred in zip(test_examples["id"], all_predictions)}
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, 'w') as f:
        f.write("id,intent\n")
        for idx, label in sorted(results.items()):
            f.write("{},{}\n".format(idx, label))

