import comet
import csv
import scipy.stats
import numpy as np
import argparse

args = argparse.ArgumentParser()
args.add_argument("checkpoint")
args = args.parse_args()

model = comet.load_from_checkpoint(args.checkpoint)
data = list(csv.DictReader(open("data/csv/dev_da.csv", "r")))
pred_y = np.array(model.predict(data, batch_size=32)["scores"]).T

data_y = [float(x["score"]) for x in data]
for layer_i, layer_y in list(enumerate(pred_y))[1:]:
    corr_gold = scipy.stats.pearsonr(data_y, layer_y).correlation
    print(
        f"Layer {layer_i:0>2}:",
        f"h={corr_gold:<5.0%}",
        end=" "
    )
    for b_layer_i, b_layer_y in list(enumerate(pred_y))[1::3]:
        corr = scipy.stats.pearsonr(b_layer_y, layer_y).correlation
        print(f"l{b_layer_i:0>2}={corr:<5.0%}", end=" ")
    print()

"""
sbatch_gpu_short "test" "python3 ../COMET-early-exit-experiments/vilem/05-score_comet.py 'lightning_logs/version_22504382/checkpoints/epoch=0-step=5864-val_avg_pearson=0.038.ckpt'"
"""