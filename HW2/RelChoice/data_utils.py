import random
import torch

def prepare_train_features(examples, args, tokenizer):
    pad_on_right = (tokenizer.padding_side == "right")
    first_sentences, second_sentences, sentence_counts, labels = [], [], [], []
    for ques, paras, label in zip(examples[args.ques_col], examples[args.para_col], examples[args.label_col]):
        if len(paras) < args.neg_num + 1:
            paras += ['' for i in range(args.neg_num + 1 - len(paras))]
        sentence_counts.append(len(paras))
        labels.append(0)    # For easier negative sampling implementation in collator.
        for i in [label] + list(range(0, label)) + list(range(label + 1, len(paras))):
            if pad_on_right:
                first_sentences.append(ques)
                second_sentences.append(paras[i])
            else:
                first_sentences.append(paras[i])
                second_sentences.append(ques)
        
    max_seq_len = min(args.max_seq_len, tokenizer.model_max_length)
    tokenized_inputs = tokenizer(
        first_sentences,
        second_sentences,
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_len,
        padding="max_length",
    )
    
    tokenized_examples = dict()
    for k, v in tokenized_inputs.items():
        tokenized_examples[k] = []
        curr = 0
        for c in sentence_counts:
            tokenized_examples[k].append(v[curr: curr + c])
            curr += c
        assert curr == len(v)
    tokenized_examples["labels"] = labels
    
    return tokenized_examples

def prepare_pred_features(examples, args, tokenizer):
    pad_on_right = (tokenizer.padding_side == "right")
    first_sentences, second_sentences, sample_indices = [], [], []
    for i, (ques, paras) in enumerate(zip(examples[args.ques_col], examples[args.para_col])):
        for p in paras:
            if pad_on_right:
                first_sentences.append(ques)
                second_sentences.append(p)
            else:
                first_sentences.append(p)
                second_sentences.append(ques)
            sample_indices.append(i)
        
    max_seq_len = min(args.max_seq_len, tokenizer.model_max_length)
    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_len,
        padding="max_length",
    )
    
    for k, v in tokenized_examples.items():
        tokenized_examples[k] = [[v_] for v_ in v]
    tokenized_examples["example_id"] = [examples["id"][sample_index] for sample_index in sample_indices]
    print(tokenized_examples)
    
    return tokenized_examples


def data_collator_with_neg_sampling(features, args):
    first = features[0]
    batch = {}

    all_para_counts = [len(f["input_ids"]) for f in features]
    select_indices = [[0] + sorted(random.sample(range(1, para_count), k=args.neg_num)) \
                        for para_count in all_para_counts]

    batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)

    for k, v in first.items():
        if k != "labels" and not isinstance(v, str):
            batch[k] = torch.stack([torch.tensor(f[k])[i] for f, i in zip(features, select_indices)])

    return batch


