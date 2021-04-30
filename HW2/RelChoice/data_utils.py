def prepare_train_features(examples, args, tokenizer):
    pad_on_right = (tokenizer.padding_side == "right")
    first_sentences, second_sentences, sentence_counts = [], [], []
    for ques, paras in zip(examples[args.ques_col], examples[args.para_col]):
        if len(paras) < 5:
            paras += ['' for i in range(5 - len(paras))]
        for p in paras:
            if pad_on_right:
                first_sentences.append(ques)
                second_sentences.append(p)
            else:
                first_sentences.append(p)
                second_sentences.append(ques)
        sentence_counts.append(len(paras))
        
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
    tokenized_examples["labels"] = examples[args.label_col]
    
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


def data_collator_with_neg_sampling():


