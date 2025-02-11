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
for score_pred, confidences, score_human in zip(pred_y, pred_conf, data_y):
    data_out.append({
        "human": score_human,
        "confidence": confidences,
        "scores": score_pred
    })

print(
    json.dumps(data_out, indent=2)
)

# TODO: this script is doing two things: evaluating early-exit conf, and evaluating last-layer with conf
# this should be split into two separate scripts

# if this is not layerwise
for layer_i in range(24):
    print(f"LAYER {layer_i}", file=sys.stderr)
    print(
        "conf corr (human)",
        scipy.stats.pearsonr([x["confidence"][layer_i] for x in data_out], [abs(x["human"]-x["scores"][layer_i]) for x in data_out]).correlation,
        file=sys.stderr,
    )
    print(
        "conf corr (last)",
        scipy.stats.pearsonr([x["confidence"][layer_i] for x in data_out], [abs(x["scores"][-1]-x["scores"][layer_i]) for x in data_out]).correlation,
        file=sys.stderr,
    )
    print(
        "human corr",
        scipy.stats.pearsonr([x["human"] for x in data_out], [x["scores"][layer_i] for x in data_out]).correlation,
        file=sys.stderr,
    )

"""
sbatch_gpu_big_short "eval_beryllium_conf" "python3 ../COMET-early-exit-experiments/vilem/05c-score_earlyexit_conf.py 'lightning_logs/version_22504386/checkpoints/epoch=4-step=29320-val_avg_pearson=0.259.ckpt'"
sbatch_gpu_big_short "eval_nitrogen_conf" "python3 ../COMET-early-exit-experiments/vilem/05c-score_earlyexit_conf.py 'lightning_logs/version_22525433/checkpoints/epoch=4-step=29320-val_avg_pearson=0.225.ckpt'"
sbatch_gpu_big_short "eval_lithium_conf" "python3 ../COMET-early-exit-experiments/vilem/05c-score_earlyexit_conf.py 'lightning_logs/version_22504384/checkpoints/epoch=4-step=29320-val_avg_pearson=0.255.ckpt'"
sbatch_gpu_big_short "eval_oxygen_conf" "python3 ../COMET-early-exit-experiments/vilem/05c-score_earlyexit_conf.py 'lightning_logs/version_22525435/checkpoints/epoch=4-step=29320-val_avg_pearson=0.254.ckpt'"
"""