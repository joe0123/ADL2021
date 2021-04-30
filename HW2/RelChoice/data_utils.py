def prepare_train_features(examples, args, tokenizer):
    pad_on_right = (tokenizer.padding_side == "right")
    first_sentences, second_sentences, sentence_counts, labels = [], [], [], []
    for ques, paras, label in zip(examples[args.ques_col], examples[args.para_col], examples[args.label_col]):
        if len(paras) < args.neg_num + 1:
            paras += ['' for i in range(args.neg_num + 1 - len(paras))]
        sentence_counts.append(len(paras))
        labels.append(0)    # For easier negative sampling implementation in collator.
        for i in [label] + range(0, label) + range(label + 1, len(params)):
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
    
    tokenized_examples["example_id"] = [examples["id"][sample_index] for sample_index in sample_indices]
    
    return tokenized_examples


def data_collator_with_neg_sampling(features, args):
    if not isinstance(features[0], (dict, BatchEncoding)):
        features = [vars(f) for f in features]

    first = features[0]
    batch = {}

    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


