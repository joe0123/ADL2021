import os
import json
import argparse
import numpy as np
import re

np.random.seed(1114)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infiles", nargs='+', type=str)
    parser.add_argument("-o", "--outfile_prefix", required=True, type=str)
    parser.add_argument("-d", "--outdir", default="task_data", type=str)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    
    with open(os.path.join(args.outdir, args.outfile_prefix + ".json"), 'w') as wf:
        for i, infile in enumerate(args.infiles):
            with open(infile, 'r') as rf:
                data = json.load(rf)
            for d in data:
                print(json.dumps(d), file=wf)
