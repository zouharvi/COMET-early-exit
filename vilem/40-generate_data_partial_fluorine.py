import argparse
import json
import collections
import csv
import numpy as np
import random
from utils import FERTILITY

r_len = random.Random(0)

args = argparse.ArgumentParser()
args.add_argument("data_in")
args.add_argument("data_out")
args = args.parse_args()


with open(args.data_in) as f:
    data = [json.loads(line) for line in f]

def random_sub(tgt):
    prefix = r_len.choice([0.0, 0.05, 0.1, 0.15, 0.2])
    suffix = r_len.choice([0.0, 0.05, 0.1, 0.15, 0.2])
    return tgt[int(prefix*len(tgt)):int((1-suffix)*len(tgt))]

data_out = []
for line in data:
    data_out.append({
        "src": line["src"],
        "mt": random_sub(line["tgt"]),
        "score": line["score"],
    })


with open(args.data_out, "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "mt", "score"])
    writer.writeheader()
    writer.writerows(data_out)

"""
python3 vilem/40-generate_data_partial_fluorine.py data/jsonl/train.jsonl data/csv/train_fluorine.csv
python3 vilem/40-generate_data_partial_fluorine.py data/jsonl/dev.jsonl data/csv/dev_fluorine.csv
python3 vilem/40-generate_data_partial_fluorine.py data/jsonl/test.jsonl data/csv/test_fluorine.csv

rsync -azP data/csv/*_fluorine.csv euler:/cluster/work/sachan/vilem/COMET-early-exit/data/csv/
"""