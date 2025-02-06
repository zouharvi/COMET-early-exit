import argparse
import json
import csv
from utils import FERTILITY

args = argparse.ArgumentParser()
args.add_argument("data_in")
args.add_argument("data_out")
args = args.parse_args()

with open(args.data_in) as f:
    data = [json.loads(line) for line in f]

data_out = []
for line in data:
    data_out.append({
        "src": line["src"],
        "mt": line["tgt"][:int(FERTILITY[line["langs"]]*len(line["src"])/2)],
        "score": line["score"],
    })


with open(args.data_out, "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "mt", "score"])
    writer.writeheader()
    writer.writerows(data_out)

"""
python3 vilem/08-generate_data_partial.py data/jsonl/train.jsonl data/csv/train_partial.csv
python3 vilem/08-generate_data_partial.py data/jsonl/dev.jsonl data/csv/dev_partial.csv
python3 vilem/08-generate_data_partial.py data/jsonl/test.jsonl data/csv/test_partial.csv
"""