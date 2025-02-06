import comet
import csv
import argparse
import json

args = argparse.ArgumentParser()
args.add_argument("checkpoint")
args.add_argument("--mc-dropout", type=int, default=0)
args = args.parse_args()

model = comet.load_from_checkpoint(args.checkpoint)
data = list(csv.DictReader(open("data/csv/test_da.csv", "r")))
out = model.predict(data, batch_size=64, mc_dropout=args.mc_dropout)
pred_y = out["scores"]

with open("logs/12b-score_mcdropout.json", "w") as f:
    json.dump(
        out,
        f,
        indent=2
    )

data_out = []
data_y = [float(x["score"]) for x in data]
for score, mcd_score, mcd_std, mcd_mad, human in zip(
    out["scores"],
    out["metadata"]["mcd_scores"],
    out["metadata"]["mcd_std"],
    out["metadata"]["mcd_mad"],
    data_y
):
    data_out.append({
        "human": human,
        "score": mcd_score,
        "mcd_score": mcd_score,
        "mcd_std": mcd_std,
        "mcd_mad": mcd_mad,
    })

print(
    json.dumps(data_out, indent=2)
)

"""
sbatch_gpu_big_short "eval_helium_mcd_02"  "python3 ../COMET-early-exit-experiments/vilem/12b-score_mcdropout.py 'lightning_logs/version_22504094/checkpoints/epoch=4-step=29320-val_pearson=0.419.ckpt' --mc-dropout 02"
sbatch_gpu_big_short "eval_helium_mcd_05"  "python3 ../COMET-early-exit-experiments/vilem/12b-score_mcdropout.py 'lightning_logs/version_22504094/checkpoints/epoch=4-step=29320-val_pearson=0.419.ckpt' --mc-dropout 05"
sbatch_gpu_big       "eval_helium_mcd_10"  "python3 ../COMET-early-exit-experiments/vilem/12b-score_mcdropout.py 'lightning_logs/version_22504094/checkpoints/epoch=4-step=29320-val_pearson=0.419.ckpt' --mc-dropout 10"
sbatch_gpu_big       "eval_helium_mcd_50"  "python3 ../COMET-early-exit-experiments/vilem/12b-score_mcdropout.py 'lightning_logs/version_22504094/checkpoints/epoch=4-step=29320-val_pearson=0.419.ckpt' --mc-dropout 50"
sbatch_gpu_big       "eval_helium_mcd_100" "python3 ../COMET-early-exit-experiments/vilem/12b-score_mcdropout.py 'lightning_logs/version_22504094/checkpoints/epoch=4-step=29320-val_pearson=0.419.ckpt' --mc-dropout 100"
"""