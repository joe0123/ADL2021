import os
import sys
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
    CONFIG_MAPPING,
    MODEL_MAPPING,
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
    parser.add_argument("--infer_file", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=384)
    parser.add_argument("--config_name", type=str)
    parser.add_argument("--tokenizer_name", type=str)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--beam", action="store_true")
    parser.add_argument("--pred_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--epoch_num", type=int, default=10)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--sched_type", type=SchedulerType, default="linear", choices=["linear", "cosine", "constant"])
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--log_steps", type=int, default=300)
    parser.add_argument("--eval_steps", type=int, default=1500)
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
    logger.addHandler(logging.FileHandler(os.path.join(args.ckpt_dir, "log")))
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
    raw_datasets = load_dataset("json", data_files={"infer": args.infer_file})
    
# Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name:
        config = AutoConfig.from_pretrained(args.model_name)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    elif args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name:
        model = AutoModelForQuestionAnswering.from_pretrained(args.model_name, config=config)
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForQuestionAnswering.from_config(config)
    
    print(tokenizer.model_max_length)
    exit()
# Preprocessing the datasets
    cols = raw_datasets["train"].column_names
    args.ques_col, args.context_col, args.ans_col = "question", "context", "answers"
# Create Training Features
    train_examples = raw_datasets["train"]
    #train_examples = train_examples.select(range(10))
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
    if args.valid_file:
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
    
# Prepare everything with our accelerator.
    if args.valid_file:
        model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, valid_dataloader
        )
    else:
        model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader
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

# Train!
    total_train_batch_size = args.train_batch_size * accelerator.num_processes * args.grad_accum_steps
    logger.info("\n******** Running training ********")
    logger.info(f"Num examples = {len(train_dataset)}")
    logger.info(f"Num Epochs = {args.epoch_num}")
    logger.info(f"Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"Total train batch size (w/ parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"Gradient Accumulation steps = {args.grad_accum_steps}")
    logger.info(f"Total optimization steps = {args.max_update_steps}")
    
    max_valid_em = 0
    for epoch in range(args.epoch_num):
        logger.info("\nEpoch {:02d} / {:02d}".format(epoch + 1, args.epoch_num))
        total_loss = 0
        for step, data in enumerate(train_dataloader, 1):
            model.train()
            outputs = model(**data)
            loss = outputs.loss
            total_loss += loss.item()
            if len(train_dataloader) - step + 1 < args.grad_accum_steps:
                loss = loss / (len(train_dataloader) % args.grad_accum_steps)
            else:
                loss = loss / args.grad_accum_steps
            accelerator.backward(loss)
            
        # Update model parameters
            if step % args.grad_accum_steps == 0 or step == len(train_dataloader):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
        # Log train loss
            if step % args.log_steps == 0 or step == len(train_dataloader):
                logger.info("Train | Loss: {:.5f}".format(total_loss / step))
        # Evaluate!
            if args.valid_file and (step % args.eval_steps == 0 or step == len(train_dataloader)):
                if args.beam:
                    all_start_top_log_probs = []
                    all_start_top_index = []
                    all_end_top_log_probs = []
                    all_end_top_index = []
                    all_cls_logits = []
                else:
                    all_start_logits = []
                    all_end_logits = []
                valid_dataset.set_format(type="torch", columns=["attention_mask", "input_ids", "token_type_ids"])
                model.eval()
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
                        else:
                            start_logits = outputs.start_logits
                            end_logits = outputs.end_logits
                            all_start_logits.append(accelerator.gather(start_logits).cpu().numpy())
                            all_end_logits.append(accelerator.gather(end_logits).cpu().numpy())

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
                else:
                    max_len = max([x.shape[1] for x in all_start_logits])
                    start_logits_concat = create_and_fill_np_array(all_start_logits, valid_dataset, max_len)
                    end_logits_concat = create_and_fill_np_array(all_end_logits, valid_dataset, max_len)
                    outputs_numpy = (start_logits_concat, end_logits_concat)

                valid_dataset.set_format(type=None, columns=list(valid_dataset.features.keys()))
                prediction = post_processing_function(valid_examples, valid_dataset, outputs_numpy, args, model)
                eval_result = metrics.compute(predictions=prediction.predictions, references=prediction.label_ids)
                valid_em, valid_f1 = eval_result["em"], eval_result["f1"]
                logger.info("Valid | EM: {:.5f}, F1: {:.5f}".format(valid_em, valid_f1))
                if valid_em >= max_valid_em:
                    max_valid_em = valid_em
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(args.ckpt_dir, save_function=accelerator.save)
                    logger.info("Saving config and model to {}...".format(args.ckpt_dir))
