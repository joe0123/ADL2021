import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from collections import Counter
from random import seed

from preprocess_utils import build_vocab

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main(args):
    seed(args.rand_seed)

    tags = set()
    words = Counter()
    for split in ["train", "eval"]:
        dataset_path = args.data_dir / f"{split}.json"
        dataset = json.loads(dataset_path.read_text())
        logging.info(f"Dataset loaded at {str(dataset_path.resolve())}")

        tags.update({tag for instance in dataset for tag in instance["tags"]})
        words.update([token for instance in dataset for token in instance["tokens"]])

    tag2idx = {tag: i + 1 for i, tag in enumerate(sorted(list(tags)))}
    tag2idx["[PAD]"] = 0
    tag_idx_path = args.output_dir / "tag2idx.json"
    tag_idx_path.write_text(json.dumps(tag2idx, indent=2))
    logging.info(f"Tag 2 index saved at {str(tag_idx_path.resolve())}")

    build_vocab(words, args.output_dir, args.glove_path)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="../data/slot/",
    )
    parser.add_argument(
        "--glove_path",
        type=Path,
        help="Path to Glove Embedding.",
        default="./glove.840B.300d.txt",
    )
    parser.add_argument("--rand_seed", type=int, help="Random seed.", default=14)
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to save the processed file.",
        default="../cache/slot/",
    )
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
