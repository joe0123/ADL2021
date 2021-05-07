import random
import torch

def prepare_features(examples, args, tokenizer, tag2id):
    tokenized_examples = tokenizer(examples[args.token_col], is_split_into_words=True)
    if examples.get(args.tag_col):
        labels = []
        for i, tags in enumerate(examples[args.tag_col]):
            word_ids = tokenized_examples.word_ids(batch_index=i)
            prev_word_id = None
            example_labels = []
            for word_id in word_ids:
                if word_id == None:
                    example_labels.append(-100) # Ignored when computing loss
                elif word_id != prev_word_id:
                    example_labels.append(tag2id[tags[word_id]])    # Put tag only on the first token of word
                else:
                    example_labels.append(-100)
                prev_word_id = word_id
            labels.append(example_labels)
        tokenized_examples["labels"] = labels

    return tokenized_examples

