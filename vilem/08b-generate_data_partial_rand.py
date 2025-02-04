import argparse
import json
import collections
import csv
import numpy as np
import random

r_len = random.Random(0)

args = argparse.ArgumentParser()
args.add_argument("data_in")
args.add_argument("data_out")
args = args.parse_args()

data_train = [json.loads(line) for line in open("data/jsonl/train.jsonl")]
fertility_train = collections.defaultdict(list)
for line in data_train:
    fertility_train[line["langs"]].append(len(line["tgt"])/len(line["src"]))
fertility_train = collections.defaultdict(lambda: 1, {l: np.average(v) for l,v in fertility_train.items()})


with open(args.data_in) as f:
    data = [json.loads(line) for line in f]

data_out = []
for line in data:
    data_out.append({
        "src": line["src"],
        "mt": line["tgt"][:int(r_len.choice([0.25, 0.50, 0.75, 1.00])*fertility_train[line["langs"]]*len(line["src"]))],
        "score": line["score"],
    })


with open(args.data_out, "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "mt", "score"])
    writer.writeheader()
    writer.writerows(data_out)

"""
python3 vilem/08b-generate_data_partial_rand.py data/jsonl/train.jsonl data/csv/train_partial.csv
python3 vilem/08b-generate_data_partial_rand.py data/jsonl/dev.jsonl data/csv/dev_partial.csv
python3 vilem/08b-generate_data_partial_rand.py data/jsonl/test.jsonl data/csv/test_partial.csv

rsync -azP data/csv/*_partial.csv euler:/cluster/work/sachan/vilem/COMET-early-exit/data/csv/
"""