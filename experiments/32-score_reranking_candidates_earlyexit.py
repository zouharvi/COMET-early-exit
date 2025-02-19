import comet
import json
import utils_candidates
import random
import argparse

args = argparse.ArgumentParser()
args.add_argument("--data", default="sample", choices=["sample", "beam"])
args = args.parse_args()

# TODO: use full data
data = random.sample(utils_candidates.get_data(args.data), k=500)

model_helium = comet.load_from_checkpoint("../COMET-early-exit/lightning_logs/helium2hydrogen/checkpoints/epoch=4-step=29320-val_pearson=0.419.ckpt")
model_beryllium = comet.load_from_checkpoint("../COMET-early-exit/lightning_logs/version_22504386/checkpoints/epoch=4-step=29320-val_avg_pearson=0.259.ckpt")

data_flat = [
    {
        "src": line["src"],
        "mt": tgt,
        "langs": line["langs"],
    }
    for line in data
    for tgt in line["tgts"]
]


data_out = {}
for model_name, model in [
    ("helium", model_helium),
    ("beryllium", model_beryllium),
]:
    out = model.predict(data_flat, batch_size=128)
    scores = out["scores"]
    if model_name == "helium":
        confidences = [0 for _ in range(len(scores))]
    else:
        confidences = out["confidences"]
    # reconstruct the original structure
    data_out[model_name] = [
        {
            "src": line["src"],
            "tgts": line["tgts"],
            "scores": [
                scores.pop(0)
                for _ in line["tgts"]
            ],
            "confidences": [
                confidences.pop(0)
                for _ in line["tgts"]
            ],
            "langs": line["langs"],
        }
        for line in data
    ]

print(json.dumps(data_out, ensure_ascii=False))


"""
# run me from COMET-early-exit-experiments
sbatch_gpu_big_short "score_candidates_reranking_sample" "python3 experiments/32-score_reranking_candidates_earlyexit.py --data sample"
sbatch_gpu_big_short "score_candidates_reranking_beam" "python3 experiments/32-score_reranking_candidates_earlyexit.py --data beam"
"""