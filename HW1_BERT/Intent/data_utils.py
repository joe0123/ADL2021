import random
import torch

def prepare_features(examples, args, tokenizer, intent2id):
    max_seq_len = min(args.max_seq_len, tokenizer.model_max_length)
    tokenized_examples = tokenizer(examples[args.text_col])
    if hasattr(examples, args.intent_col):
        tokenized_examples["labels"] = [intent2id[intent] for intent in examples[args.intent_col]]

    return tokenized_examples

