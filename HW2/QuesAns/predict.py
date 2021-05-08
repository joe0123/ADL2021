import os
import sys
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
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    default_data_collator,
    set_seed,
)

from data_utils import *
from pred_utils import *

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--target_dir", type=str, required=True)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--out_file", type=str, default="./results.json")
    parser.add_argument("--n_best", type=int, default=20)
    parser.add_argument("--max_ans_len", type=int, default=30)
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
    model = AutoModelForQuestionAnswering.from_pretrained(args.target_dir, config=config)

# Load and preprocess the dataset
    raw_datasets = load_dataset("json", data_files={"test": args.test_file})
    cols = raw_datasets["test"].column_names
    args.ques_col, args.context_col, args.ans_col = "question", "context", "answers"
    
    test_examples = raw_datasets["test"]
    prepare_pred_features = partial(prepare_pred_features, args=args, tokenizer=tokenizer)
    test_dataset = test_examples.map(
        prepare_pred_features,
        batched=True,
        num_proc=4,
        remove_columns=cols,
    )

# Create DataLoaders
    data_collator = default_data_collator
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.test_batch_size)
    
# Prepare everything with our accelerator.
    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )

# Test!
    logger.info("\n******** Running predicting ********")
    logger.info(f"Num test examples = {len(test_dataset)}")
    
    test_dataset.set_format(type="torch", columns=["attention_mask", "input_ids", "token_type_ids"])
    model.eval()
    all_start_logits = []
    all_end_logits = []
    for step, data in enumerate(test_dataloader):
        with torch.no_grad():
            outputs = model(**data)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            all_start_logits.append(accelerator.gather(start_logits).cpu().numpy())
            all_end_logits.append(accelerator.gather(end_logits).cpu().numpy())

    max_len = max([x.shape[1] for x in all_start_logits])
    start_logits_concat = create_and_fill_np_array(all_start_logits, test_dataset, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, test_dataset, max_len)
    outputs_numpy = (start_logits_concat, end_logits_concat)

    test_dataset.set_format(type=None, columns=list(test_dataset.features.keys()))
    predictions = post_processing_function(test_examples, test_dataset, outputs_numpy, args, model)
    results = {d["id"]: d["pred"] for d in predictions.predictions}
    
    out_dir = os.path.dirname(args.out_file)
    if len(out_dir) > 0:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
