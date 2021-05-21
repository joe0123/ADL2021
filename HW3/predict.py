import os
import sys
import json
import argparse
import logging
import math
import numpy as np
from functools import partial
from time import strftime, localtime
import datasets
from datasets import load_dataset, load_metric
from accelerate import Accelerator
import torch
from torch.utils.data.dataloader import DataLoader
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
    set_seed,
)


from data_utils import *

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--target_dir", type=str, required=True)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--out_file", type=str, default="./results.jsonl")
    parser.add_argument("--max_target_len", type=int, default=64)
    parser.add_argument("--beam_num", type=int, default=5)
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
# Parse arguments and save them.
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
    model = AutoModelForSeq2SeqLM.from_pretrained(args.target_dir, config=config)    
    
    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")


# Load and preprocess the dataset
    raw_datasets = load_dataset("json", data_files={"test": args.test_file})
    cols = raw_datasets["test"].column_names
    args.id_col, args.text_col = "id", "maintext"
    
    test_examples = raw_datasets["test"]
    #test_examples = test_examples.select(range(10))
    prepare_pred_features = partial(prepare_pred_features, args=args, tokenizer=tokenizer)
    test_dataset = test_examples.map(
        prepare_pred_features,
        batched=True,
        num_proc=4,
        remove_columns=cols,
    )
    
# Create DataLoaders
    data_collator = default_data_collator
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, 
                        batch_size=args.test_batch_size, num_workers=4)

# Prepare everything with our accelerator.
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

# Test!
    total_train_batch_size = args.train_batch_size * accelerator.num_processes * args.grad_accum_steps
    logger.info("\n******** Running training ********")
    logger.info(f"Num test examples = {len(test_dataset)}")
    
    model.eval()
    gen_kwargs = {
        "max_length": args.max_target_len,
        "num_beams": args.beam_num,
    }
    all_preds = []
    for step, data in enumerate(test_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                data["input_ids"],
                attention_mask=data["attention_mask"],
                **gen_kwargs,
            )
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            all_preds += decoded_preds
    
    with open(args.out_file, 'w') as f:
        for id_, pred in zip(test_examples[args.id_col], all_preds):
            print(json.dumps({"title": pred, "id": id_}, ensure_ascii=False), file=f)
