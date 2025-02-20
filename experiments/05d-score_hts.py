import comet_early_exit
import csv
import argparse
import json
import sys
import scipy.stats

args = argparse.ArgumentParser()
args.add_argument("checkpoint")
args = args.parse_args()

model = comet_early_exit.load_from_checkpoint(args.checkpoint)
data = list(csv.DictReader(open("data/csv/test_da.csv", "r")))
out = model.predict(data, batch_size=128)
pred_y = out[0]
pred_conf = out[1]


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
sbatch_gpu_big_short "eval_potassium_conf" "python3 ../COMET-early-exit-experiments/vilem/05d-score_hts.py '/cluster/work/sachan/vilem/COMET-zerva/lightning_logs/version_2/checkpoints/epoch=2-step=17592-val_avg_pearson=0.000.ckpt'"
"""