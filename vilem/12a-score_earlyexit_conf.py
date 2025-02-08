import comet
import csv
import argparse
import json
import sys
import scipy.stats

args = argparse.ArgumentParser()
args.add_argument("checkpoint")
args = args.parse_args()

model = comet.load_from_checkpoint(args.checkpoint)
data = list(csv.DictReader(open("data/csv/test_da.csv", "r")))
out = model.predict(data, batch_size=128)
pred_y = out["scores"]
pred_conf = out["confidences"]


data_out = []
data_y = [float(x["score"]) for x in data]
for scores, confidences, score in zip(pred_y, pred_conf, data_y):
    data_out.append({
        "human": score,
        "confidence": confidences,
        "scores": scores
    })

print(
    json.dumps(data_out, indent=2)
)

print(
    "conf corr",
    scipy.stats.pearsonr(pred_conf, [abs(x-y) for x, y in zip(data_y, pred_y)]),
    file=sys.stderr,
)
print(
    "human corr",
    scipy.stats.pearsonr(pred_y, data_y),
    file=sys.stderr,
)

"""
sbatch_gpu_big_short "eval_beryllium_conf" "python3 ../COMET-early-exit-experiments/vilem/12-score_earlyexit_conf.py 'lightning_logs/version_22504386/checkpoints/epoch=4-step=29320-val_avg_pearson=0.259.ckpt'"
sbatch_gpu_big_short "eval_lithium_conf" "python3 ../COMET-early-exit-experiments/vilem/12-score_earlyexit_conf.py 'lightning_logs/version_22504384/checkpoints/epoch=4-step=29320-val_avg_pearson=0.255.ckpt'"
sbatch_gpu_big_short "eval_hydrogen_conf" "python3 ../COMET-early-exit-experiments/vilem/12-score_earlyexit_conf.py 'lightning_logs/version_22504382/checkpoints/epoch=4-step=29320-val_avg_pearson=0.259.ckpt'"
"""