import os
import logging
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
    AdamW,
    DataCollatorWithPadding,
    SchedulerType,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
    set_seed,
)

from data_utils import *
from pred_utils import *

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--valid_file", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=384)
    parser.add_argument("--pretrained_name", type=str, required=True)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--valid_batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epoch_num", type=int, default=3)
    parser.add_argument("--grad_accum_steps", type=int, default=32)
    parser.add_argument("--sched_type", type=SchedulerType, default="linear", choices=["linear", "cosine", "constant"])
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--log_steps", type=int, default=32)
    parser.add_argument("--eval_steps", type=int, default=32)
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
    config = AutoConfig.from_pretrained(args.pretrained_name)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_name, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.pretrained_name, config=config)

    # Preprocessing the datasets
    cols = raw_datasets["train"].column_names
    args.ques_col, args.context_col, args.ans_col = "question", "context", "answers"
    # Create Training Features
    train_examples = raw_datasets["train"]
    prepare_train_features = partial(prepare_train_features, args=args, tokenizer=tokenizer)
    train_dataset = train_examples.map(
        prepare_train_features,
        batched=True,
        num_proc=4,
        remove_columns=cols,
    )

    # Create Valid Features
    if args.valid_file:
        valid_examples = raw_datasets["valid"]
        #valid_examples = valid_examples.select(range(10))
        prepare_pred_features = partial(prepare_pred_features, args=args, tokenizer=tokenizer)
        valid_dataset = valid_examples.map(
            prepare_pred_features,
            batched=True,
            num_proc=4,
            remove_columns=cols,
        )

    # Create DataLoaders
    data_collator = default_data_collator
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.train_batch_size)
    valid_dataset.set_format(type="torch", columns=["attention_mask", "input_ids", "token_type_ids"])
    valid_dataloader = DataLoader(valid_dataset, collate_fn=data_collator, batch_size=args.valid_batch_size)
    
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_gparams = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_gparams, lr=args.lr)
    
    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader
    )

    # Scheduler and math around the number of training steps.
    update_steps_per_epoch = math.ceil(len(train_dataloader) / args.grad_accum_steps)
    args.max_update_steps = args.epoch_num * update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.sched_type,
        optimizer=optimizer,
        num_warmup_steps=int(args.max_update_steps * args.warmup_ratio),
        num_training_steps=args.max_update_steps,
    )
    
    # Metrics for evaluation
    metrics = load_metric("./metrics.py")

    total_train_batch_size = args.train_batch_size * accelerator.num_processes * args.grad_accum_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.epoch_num}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w/ parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.grad_accum_steps}")
    logger.info(f"  Total optimization steps = {args.max_update_steps}")
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_update_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    for epoch in range(args.epoch_num):
        model.train()
        for step, data in enumerate(train_dataloader):
            outputs = model(**data)
            loss = outputs.loss
            loss = loss / args.grad_accum_steps
            accelerator.backward(loss)
            if step % args.grad_accum_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
            
            if completed_steps >= args.max_update_steps:
                break
    
        # intialize all lists to collect the batches
        if args.beam:
            all_start_top_log_probs = []
            all_start_top_index = []
            all_end_top_log_probs = []
            all_end_top_index = []
            all_cls_logits = []
        for step, data in enumerate(valid_dataloader):
            with torch.no_grad():
                outputs = model(**data)
                if args.beam:
                    start_top_log_probs = outputs.start_top_log_probs
                    start_top_index = outputs.start_top_index
                    end_top_log_probs = outputs.end_top_log_probs
                    end_top_index = outputs.end_top_index
                    cls_logits = outputs.cls_logits
                    all_start_top_log_probs.append(accelerator.gather(start_top_log_probs).cpu().numpy())
                    all_start_top_index.append(accelerator.gather(start_top_index).cpu().numpy())
                    all_end_top_log_probs.append(accelerator.gather(end_top_log_probs).cpu().numpy())
                    all_end_top_index.append(accelerator.gather(end_top_index).cpu().numpy())
                    all_cls_logits.append(accelerator.gather(cls_logits).cpu().numpy())

        if args.beam:
            max_len = max([x.shape[1] for x in all_end_top_log_probs])  # Get the max_length of the tensor
            start_top_log_probs_concat = create_and_fill_np_array(all_start_top_log_probs, valid_dataset, max_len)
            start_top_index_concat = create_and_fill_np_array(all_start_top_index, valid_dataset, max_len)
            end_top_log_probs_concat = create_and_fill_np_array(all_end_top_log_probs, valid_dataset, max_len)
            end_top_index_concat = create_and_fill_np_array(all_end_top_index, valid_dataset, max_len)
            all_cls_logits = np.concatenate(all_cls_logits, axis=0)

            outputs_numpy = (
                start_top_log_probs_concat,
                start_top_index_concat,
                end_top_log_probs_concat,
                end_top_index_concat,
                all_cls_logits,
            )
        valid_dataset.set_format(type=None, columns=list(valid_dataset.features.keys()))
        prediction = post_processing_function(valid_examples, valid_dataset, outputs_numpy, args, model)
        eval_metric = metrics.compute(predictions=prediction.predictions, references=prediction.label_ids)
        logger.info(f"Evaluation metrics: {eval_metric}")
