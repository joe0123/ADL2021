import argparse
import os
import numpy as np
import pandas as pd
from gensim.models import word2vec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", default="../data/train.csv", type=str)
    parser.add_argument("--valid_data", default="../data/dev.csv", type=str)
    parser.add_argument("--test_data", default="../data/test.csv", type=str)
    args = parser.parse_args()

    print("Loading data...", flush=True)
    x = []
    for case in ["train", "valid", "test"]: 
        df = pd.read_csv(getattr(args, "{}_data".format(case)))
        x += [st.split() for st in df.text.tolist()]
    
    print("Word2Vec...", flush=True)
    model = word2vec.Word2Vec(x, size=256, window=5, min_count=10, workers=12, iter=10, sg=1)
    
    print("Saving model...", flush=True)
    model.save("w2v.model")

