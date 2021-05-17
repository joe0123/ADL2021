import numpy as np
np.random.seed(1114)

with open("train.jsonl", 'r') as f:
    data = f.readlines()

all_indices = np.random.permutation(np.arange(len(data)))
cut = int(len(data) * 0.2)
with open("train_split.jsonl", 'w') as f:
    for i in all_indices[cut:].tolist():
        f.write(data[i])
with open("valid_split.jsonl", 'w') as f:
    for i in all_indices[:cut].tolist():
        f.write(data[i])
