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

# if this is not layerwise
print(
    "conf corr (human)",
    scipy.stats.pearsonr([x["confidence"] for x in data_out], [abs(x["human"]-x["scores"]) for x in data_out]).correlation,
    file=sys.stderr,
)
print(
    "human corr",
    scipy.stats.pearsonr([x["human"] for x in data_out], [x["scores"] for x in data_out]).correlation,
    file=sys.stderr,
)

"""
sbatch_gpu_big_short "eval_magnesium_conf" "python3 ../COMET-early-exit-experiments/vilem/05b-score_conf.py 'lightning_logs/version_22525447/checkpoints/epoch=4-step=29320-val_pearson=0.405.ckpt'"
sbatch_gpu_big_short "eval_phosphorus_conf" "python3 ../COMET-early-exit-experiments/vilem/05b-score_conf.py 'lightning_logs/version_22525456/checkpoints/epoch=4-step=29320-val_pearson=0.419.ckpt'"
sbatch_gpu_big_short "eval_sulfur_conf" "python3 ../COMET-early-exit-experiments/vilem/05b-score_conf.py 'lightning_logs/version_22525457/checkpoints/epoch=4-step=29320-val_pearson=0.415.ckpt'"
sbatch_gpu_big_short "eval_chlorine_conf" "python3 ../COMET-early-exit-experiments/vilem/05b-score_conf.py 'lightning_logs/version_22525458/checkpoints/epoch=4-step=29320-val_pearson=0.423.ckpt'"
sbatch_gpu_big_short "eval_argon_conf" "python3 ../COMET-early-exit-experiments/vilem/05b-score_conf.py 'lightning_logs/version_22525459/checkpoints/epoch=4-step=29320-val_pearson=0.425.ckpt'"
"""