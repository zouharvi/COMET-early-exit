import comet_early_exit
import csv
import scipy.stats
import numpy as np
import argparse
import json
import sys

args = argparse.ArgumentParser()
args.add_argument("checkpoint")
args.add_argument("--print", action="store_true")
args = args.parse_args()

model = comet_early_exit.load_from_checkpoint(args.checkpoint)
data = list(csv.DictReader(open("data/csv/test_da.csv", "r")))
pred_y = np.array(model.predict(data, batch_size=128)["scores"]).T

data_out = []
data_y = [float(x["score"]) for x in data]
for layer_i, layer_y in list(enumerate(pred_y)):
    corr_gold = scipy.stats.pearsonr(data_y, layer_y).correlation
    data_out.append({
        "layer": layer_i,
        "corr_human": corr_gold,
        "corr": [scipy.stats.pearsonr(layer_y, b_layer_y).correlation for b_layer_y in pred_y]
    })
    print(
        f"Layer {layer_i:0>2}:",
        f"h={corr_gold:<5.0%}",
        end=" ",
        file=sys.stderr,
    )
    for b_layer_i, b_layer_y in list(enumerate(pred_y))[1::3]:
        corr = scipy.stats.pearsonr(b_layer_y, layer_y).correlation
        print(f"l{b_layer_i:0>2}={corr:<5.0%}", end=" ", file=sys.stderr)
    print(file=sys.stderr)

print(
    json.dumps(data_out, indent=2)
)

"""
sbatch_gpu_big_short "eval_oxygen" "python3 ../COMET-early-exit-experiments/vilem/05a-score_base.py 'lightning_logs/version_22525435/checkpoints/epoch=4-step=29320-val_avg_pearson=0.254.ckpt'"
sbatch_gpu_big_short "eval_helium2hydrogen" "python3 ../COMET-early-exit-experiments/vilem/05a-score_base.py 'lightning_logs/helium2hydrogen/checkpoints/epoch=4-step=29320-val_pearson=0.419.ckpt'"
sbatch_gpu_big_short "eval_nitrogen" "python3 ../COMET-early-exit-experiments/vilem/05a-score_base.py 'lightning_logs/version_22525433/checkpoints/epoch=4-step=29320-val_avg_pearson=0.225.ckpt'"
sbatch_gpu_big_short "eval_beryllium" "python3 ../COMET-early-exit-experiments/vilem/05a-score_base.py 'lightning_logs/version_22504386/checkpoints/epoch=4-step=29320-val_avg_pearson=0.259.ckpt'"

sbatch_gpu_big_short "eval_lithium" "python3 ../COMET-early-exit-experiments/vilem/05a-score_base.py 'lightning_logs/version_22504384/checkpoints/epoch=4-step=29320-val_avg_pearson=0.255.ckpt'"
sbatch_gpu_big_short "eval_hydrogen" "python3 ../COMET-early-exit-experiments/vilem/05a-score_base.py 'lightning_logs/version_22504382/checkpoints/epoch=4-step=29320-val_avg_pearson=0.259.ckpt'"
"""