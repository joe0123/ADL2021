import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--valid_file", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=384)
    parser.add_argument("--pretrained_model", type=str, required=True)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--valid_batch_size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epoch_num", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--grad_accum_steps", type=int, default=32)
    parser.add_argument("--sched_type", type=SchedulerType, default="linear", choices=["linear", "cosine", "constant"])
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--ckpt_dir", type=str, default="./ckpt" help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=14, help="A seed for reproducible training.")
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
    raw_datasets = load_dataset("json", {"train": args.train_file, "valid": args.valid_file})
    
    # Load pretrained model and tokenizer
    config = XLNetConfig.from_pretrained(args.pretrained_name)
    tokenizer = XLNetTokenizerFast.from_pretrained(args.pretrained_name)
    model = XLNetForQuestionAnswering.from_pretrained(args.pretrained_name, config=config)

    # Preprocessing the datasets
    ques_col, context_col, ans_col = "question", "context", "answers"
