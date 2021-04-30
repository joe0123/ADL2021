import os
import json
import argparse
import numpy as np

np.random.seed(1114)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--ques_data", required=True, type=str)
    parser.add_argument("-c", "--context_data", required=True, type=str)
    parser.add_argument("-o", "--outfile_prefix", required=True, type=str)
    parser.add_argument("-d", "--outdir", default="task_data", type=str)
    parser.add_argument("-s", "--split_ratio", default=0, type=float)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    
    with open(args.ques_data, 'r') as f:
        ques_data = json.load(f)
    with open(args.context_data, 'r') as f:
        context_data = json.load(f)
    
    if args.split_ratio > 0:
        all_indices = np.random.permutation(np.arange(len(ques_data)))
        cut = int(len(ques_data) * args.split_ratio)
        split_indices = [all_indices[cut:].tolist(), all_indices[:cut].tolist()]
    else:
        split_indices = [np.arange(len(ques_data)).tolist()]

    
    for si, ids in enumerate(split_indices):
        with open(os.path.join(args.outdir, args.outfile_prefix + "_{}.json".format(si)), 'w') as f:
            for i in ids:
                q_data = ques_data[i]
                paragraphs = []
                for pi, p in enumerate(q_data["paragraphs"]):
                    paragraphs.append(context_data[p])
                    if p == q_data["relevant"]:
                        rel = pi
                data = {"id": q_data["id"],
                        "question": q_data["question"],
                        "paragraphs": paragraphs,
                        "relevant": rel}
                print(json.dumps(data, ensure_ascii=False), file=f)
