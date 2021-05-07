import random
import torch

def prepare_features(examples, args, tokenizer, intent2id):
    tokenized_examples = tokenizer(examples[args.text_col])
    if examples.get(args.intent_col):
        tokenized_examples["labels"] = [intent2id[intent] for intent in examples[args.intent_col]]
    return tokenized_examples

