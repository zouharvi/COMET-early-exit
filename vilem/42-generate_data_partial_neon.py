import argparse
import collections
import json
import csv
import random
import numpy as np
r_len = np.random.RandomState(0)

args = argparse.ArgumentParser()
args.add_argument("data_in")
args.add_argument("data_out")
args = args.parse_args()


with open(args.data_in) as f:
    data = [json.loads(line) for line in f]

start_words_in_lang = collections.defaultdict(list)
for line in data:
    if not line["tgt"]:
        continue
    start_word = line["tgt"].split()[0]
    if len(start_word) < 15:
        start_words_in_lang[line["langs"]].append(start_word)


def random_sub(tgt: str, lang: str):
    choice = r_len.choice(
        ["nothing", "prefix remove", "suffix remove", "lowercase start", "prefix symbol", "suffix symbol", "prefix add"],
        1,
        p=[0.945, 0.01, 0.01, 0.01, 0.01, 0.01, 0.005]
    )[0]
    if choice == "prefix symbol":
        return r_len.choice([". ", "- ", ", "]) + tgt
    if choice == "suffix symbol":
        return tgt.removesuffix(".").removesuffix("?")
    
    if not tgt:
        return tgt
    
    if choice == "prefix add":
        return r_len.choice(start_words_in_lang[lang], 1)[0] + " " + tgt

    first_word = tgt.split()[0]
    first_word_short = len(first_word) < 15
    if choice == "prefix remove" and first_word_short:
        return tgt[len(first_word):].strip()
    if choice == "lowercase start" and first_word_short:
        return tgt.replace(first_word, first_word.lower(), 1)
    last_word = tgt.split()[-1]
    last_word_short = len(last_word) < 15
    if choice == "suffix remove" and last_word_short:
        return tgt[:-len(last_word)].strip()
    
    return tgt

data_out = []
for line in data:
    data_out.append({
        "src": line["src"],
        "mt": random_sub(line["tgt"], line["langs"]),
        "score": line["score"],
    })


with open(args.data_out, "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "mt", "score"])
    writer.writeheader()
    writer.writerows(data_out)

"""
python3 vilem/42-generate_data_partial_neon.py data/jsonl/train.jsonl data/csv/train_neon.csv
python3 vilem/42-generate_data_partial_neon.py data/jsonl/dev.jsonl data/csv/dev_neon.csv
python3 vilem/42-generate_data_partial_neon.py data/jsonl/test.jsonl data/csv/test_neon.csv

rsync -azP data/csv/*_neon.csv euler:/cluster/work/sachan/vilem/COMET-early-exit/data/csv/
"""