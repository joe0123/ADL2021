def prepare_train_features(args, examples, tokenizer, ques_col, context_col, ans_col):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    pad_on_right = (tokenizer.padding_side == "right")
    max_seq_len = min(args.max_seq_len, tokenizer.model_max_length)
    tokenized_examples = tokenizer(
        examples[ques_col if pad_on_right else context_col],
        examples[context_col if pad_on_right else ques_col],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_len,
        stride=args.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        return_token_type_ids=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mappings = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mappings = tokenized_examples.pop("offset_mapping")
    # The special tokens will help us build the p_mask (which indicates the tokens that can't be in answers).
    special_ids = tokenized_examples.pop("special_tokens_mask")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples["is_impossible"] = []
    tokenized_examples["cls_index"] = []
    tokenized_examples["p_mask"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        tokenized_examples["cls_index"].append(cls_index)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples["token_type_ids"][i]
        for k, s in enumerate(special_tokens[i]):
            if s:
                sequence_ids[k] = 3
        context_idx = 1 if pad_on_right else 0

        # Build the p_mask: non special tokens and context gets 0.0, the others get 1.0.
        # The cls token gets 1.0 too (for predictions of empty answers).
        tokenized_examples["p_mask"].append(
            [0.0 if (not special_ids[i][k] and s == context_idx) or k == cls_index else 1.0 \
                for k, s in enumerate(sequence_ids)]
        )

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples[ans_col][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answers"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
            tokenized_examples["is_impossible"].append(1.0)
        else:
            # Start/end character index of the answer in the text.
            # TODO
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != context_idx:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != context_idx:
                token_end_index -= 1
            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
                tokenized_examples["is_impossible"].append(1.0)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)
                tokenized_examples["is_impossible"].append(0.0)

    return tokenized_examples
