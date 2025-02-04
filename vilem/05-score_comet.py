import comet
import csv
import scipy.stats
import numpy as np
import argparse
import json

args = argparse.ArgumentParser()
args.add_argument("checkpoint")
args = args.parse_args()

model = comet.load_from_checkpoint(args.checkpoint)
data = list(csv.DictReader(open("data/csv/dev_da.csv", "r")))
pred_y = np.array(model.predict(data, batch_size=32)["scores"]).T


data_out = []
data_y = [float(x["score"]) for x in data]
for layer_i, layer_y in list(enumerate(pred_y))[1:]:
    corr_gold = scipy.stats.pearsonr(data_y, layer_y).correlation
    data_out.append({
        "layer": layer_i,
        "corr_human": corr_gold,
        "corr": [scipy.stats.pearsonr(layer_y, b_layer_y).correlation for b_layer_y in pred_y]
    })
    print(
        f"Layer {layer_i:0>2}:",
        f"h={corr_gold:<5.0%}",
        end=" "
    )
    for b_layer_i, b_layer_y in list(enumerate(pred_y))[1::3]:
        corr = scipy.stats.pearsonr(b_layer_y, layer_y).correlation
        print(f"l{b_layer_i:0>2}={corr:<5.0%}", end=" ")
    print()

print(
    json.dumps(data_out, indent=2)
)

"""
sbatch_gpu_short "eval_beryllium" "python3 ../COMET-early-exit-experiments/vilem/05-score_comet.py 'lightning_logs/version_22504386/checkpoints/epoch=4-step=29320-val_avg_pearson=0.259.ckpt'"

sbatch_gpu_short "eval_hydrogen" "python3 ../COMET-early-exit-experiments/vilem/05-score_comet.py 'lightning_logs/version_22504382/checkpoints/epoch=4-step=29320-val_avg_pearson=0.259.ckpt'"
"""