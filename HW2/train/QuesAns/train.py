import os
import logging
import argparse
import logging
from functools import partial
from time import strftime, localtime
import datasets
from datasets import load_dataset, load_metric
from accelerate import Accelerator
import transformers
from transformers import (
    AdamW,
    DataCollatorWithPadding,
    EvalPrediction,
    SchedulerType,
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLNetTokenizerFast,
    default_data_collator,
    get_scheduler,
    set_seed,
)

from data_utils import *

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--valid_file", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=384)
    parser.add_argument("--pretrained_name", type=str, required=True)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--valid_batch_size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epoch_num", type=int, default=3)
    parser.add_argument("--grad_accum_steps", type=int, default=32)
    parser.add_argument("--sched_type", type=SchedulerType, default="linear", choices=["linear", "cosine", "constant"])
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--ckpt_dir", type=str, default="./ckpt")
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--n_best", type=int, default=20)
    parser.add_argument("--max_ans_len", type=int, default=30)
    args = parser.parse_args()
    
    args.ckpt_dir = os.path.join(args.ckpt_dir, strftime("%m%d-%H%M", localtime()))
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    return args

if __name__ == "__main__":
    args = parse_args()

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
    
    # Load datasets
    if args.valid_file:
        raw_datasets = load_dataset("json", data_files={"train": args.train_file, "valid": args.valid_file})
    else:
        raw_datasets = load_dataset("json", data_files={"train": args.train_file})
    
    # Load pretrained model and tokenizer
    config = XLNetConfig.from_pretrained(args.pretrained_name)
    tokenizer = XLNetTokenizerFast.from_pretrained(args.pretrained_name)
    model = XLNetForQuestionAnswering.from_pretrained(args.pretrained_name, config=config)

    # Preprocessing the datasets
    cols = raw_datasets["train"].column_names
    ques_col, context_col, ans_col = "question", "context", "answers"
    # Create Training Features
    train_dataset = raw_datasets["train"]
    prepare_train_features = partial(prepare_train_features, 
                                    args=args, tokenizer=tokenizer, 
                                    ques_col=ques_col, context_col=context_col, ans_col=ans_col)
    train_dataset = train_dataset.map(
        prepare_train_features,
        batched=True,
        num_proc=4,
        remove_columns=cols,
    )

    # Create Valid Features
    if args.valid_file:
        valid_dataset = raw_datasets["valid"]
        prepare_valid_features = partial(prepare_valid_features, 
                                        args=args, tokenizer=tokenizer, 
                                        ques_col=ques_col, context_col=context_col, ans_col=ans_col)
        valid_dataset = valid_dataset.map(
            prepare_valid_features,
            batched=True,
            num_proc=4,
            remove_columns=cols,
        )

    # Create DataLoaders
    data_collator = default_data_collator
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.train_batch_size)
    valid_dataset.set_format(type="torch", columns=["attention_mask", "input_ids", "token_type_ids"])
    valid_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.valid_batch_size)

