def prepare_train_features(examples, args, tokenizer):
    pad_on_right = (tokenizer.padding_side == "right")
    max_seq_len = min(args.max_seq_len, tokenizer.model_max_length)
    tokenized_examples = tokenizer(
        examples[args.ques_col if pad_on_right else args.context_col],
        examples[args.context_col if pad_on_right else args.ques_col],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_len,
        stride=args.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        return_token_type_ids=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    special_tokens = tokenized_examples.pop("special_tokens_mask")
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    if args.beam:
        tokenized_examples["is_impossible"] = []
        tokenized_examples["cls_index"] = []
        tokenized_examples["p_mask"] = []
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        if args.beam:
            tokenized_examples["cls_index"].append(cls_index)

        sequence_ids = tokenized_examples.sequence_ids(i)
        #if args.beam:
        #    for k, s in enumerate(special_tokens[i]):
        #        if s:
        #            sequence_ids[k] = 3
        context_idx = 1 if pad_on_right else 0

        if args.beam:
            tokenized_examples["p_mask"].append(
                [0.0 if (not special_tokens[i][k] and s == context_idx) or k == cls_index else 1.0 \
                    for k, s in enumerate(sequence_ids)]
            )

        sample_index = sample_mapping[i]
        answers = examples[args.ans_col][sample_index]
        if len(answers) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
            if args.beam:
                tokenized_examples["is_impossible"].append(1.0)
        else: # TODO multiple answers?
            char_start_index = answers[0]["start"]
            char_end_index = char_start_index + len(answers[0]["text"])

            token_start_index = 0
            while sequence_ids[token_start_index] != context_idx:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != context_idx:
                token_end_index -= 1
            
            if not (offsets[token_start_index][0] <= char_start_index and offsets[token_end_index][1] >= char_end_index):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
                if args.beam:
                    tokenized_examples["is_impossible"].append(1.0)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= char_start_index:
                    token_start_index += 1
                token_start_index -= 1
                tokenized_examples["start_positions"].append(token_start_index)
                while offsets[token_end_index][1] >= char_end_index:
                    token_end_index -= 1
                token_end_index += 1
                tokenized_examples["end_positions"].append(token_end_index)
                if args.beam:
                    tokenized_examples["is_impossible"].append(0.0)
                
                #if offsets[token_start_index][0] != char_start_index or offsets[token_end_index][1] != char_end_index:
                    #print("OhOhOh", sample_index, offsets[token_start_index], offsets[token_end_index], examples[args.context_col][sample_index][offsets[token_start_index][0]: offsets[token_end_index][1]], answers[0]["text"])

    return tokenized_examples

def prepare_pred_features(examples, args, tokenizer):
    pad_on_right = (tokenizer.padding_side == "right")
    max_seq_len = min(args.max_seq_len, tokenizer.model_max_length)
    tokenized_examples = tokenizer(
        examples[args.ques_col if pad_on_right else args.context_col],
        examples[args.context_col if pad_on_right else args.ques_col],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_len,
        stride=args.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        return_token_type_ids=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    special_tokens = tokenized_examples.pop("special_tokens_mask")
    tokenized_examples["example_id"] = []
    if args.beam:
        tokenized_examples["cls_index"] = []
        tokenized_examples["p_mask"] = []
    for i, input_ids in enumerate(tokenized_examples["input_ids"]):
        cls_index = input_ids.index(tokenizer.cls_token_id)
        if args.beam:
            tokenized_examples["cls_index"].append(cls_index)

        sequence_ids = tokenized_examples.sequence_ids(i)
        #if args.beam:
        #    for k, s in enumerate(special_tokens[i]):
        #        if s:
        #            sequence_ids[k] = 3
        context_idx = 1 if pad_on_right else 0

        if args.beam:
            tokenized_examples["p_mask"].append(
                [0.0 if (not special_tokens[i][k] and s == context_idx) or k == cls_index else 1.0
                    for k, s in enumerate(sequence_ids)]
            )

        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_idx else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])]

    return tokenized_examples


