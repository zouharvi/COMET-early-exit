import comet
import json
from utils import FERTILITY
import utils_candidates
import random
import argparse

args = argparse.ArgumentParser()
args.add_argument("--data", default="sample", choices=["sample", "beam"])
args = args.parse_args()

# TODO: use full
data = random.sample(utils_candidates.get_data(args.data), k=500)

model_helium = comet.load_from_checkpoint("../COMET-early-exit/lightning_logs/version_22504094/checkpoints/epoch=4-step=29320-val_pearson=0.419.ckpt")
model_carbon = comet.load_from_checkpoint("../COMET-early-exit/lightning_logs/version_22525421/checkpoints/epoch=4-step=29320-val_pearson=0.380.ckpt")

data_flat = [
    {
        "src": line["src"],
        "mt": tgt[:int(p*FERTILITY[line["langs"]]*len(line["src"]))],
        "langs": line["langs"],
    }
    for line in data
    for tgt in line["tgts"]
    for p in [0.25, 0.50, 0.75, 10.00]
]

# NOTE: use 1_000% (10x) to make sure the whole translation is in

data_out = {}
for model_name, model in [
    ("helium", model_helium),
    ("carbon", model_carbon),
]:
    scores = model.predict(data_flat, batch_size=128)["scores"]
    # reconstruct the original structure
    data_out[model_name] = [
        {
            "src": line["src"],
            "tgts": line["tgts"],
            "scores": [
                {
                    p: scores.pop(0)
                    for p in [0.25, 0.50, 0.75, 10.00]
                }
                for _ in line["tgts"]
            ],
            "langs": line["langs"],
        }
        for line in data
    ]

print(json.dumps(data_out, ensure_ascii=False))

"""
sbatch_gpu_big_short "score_candidates_partial_sample" "python3 ../COMET-early-exit-experiments/vilem/30-score_partial_candidates.py --data sample"
sbatch_gpu_big_short "score_candidates_partial_beam" "python3 ../COMET-early-exit-experiments/vilem/30-score_partial_candidates.py --data beam"
"""