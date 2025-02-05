import comet
import csv
import scipy.stats
import json
import collections
import numpy as np

data_orig = [json.loads(line) for line in open("data/jsonl/dev.jsonl")]

model_orig = comet.load_from_checkpoint("lightning_logs/version_22504094/checkpoints/epoch=4-step=29320-val_pearson=0.419.ckpt")
model_boron = comet.load_from_checkpoint("lightning_logs/version_22525419/checkpoints/epoch=4-step=29320-val_pearson=0.377.ckpt")
model_carbon = comet.load_from_checkpoint("lightning_logs/version_22525421/checkpoints/epoch=4-step=29320-val_pearson=0.380.ckpt")

data_train = [json.loads(line) for line in open("data/jsonl/train.jsonl")]
fertility_train = collections.defaultdict(list)
for line in data_train:
    fertility_train[line["langs"]].append(len(line["tgt"])/len(line["src"]))
fertility_train = collections.defaultdict(lambda: 1, {l: np.average(v) for l,v in fertility_train.items()})

data_all = {
    p: [
        {
            "src": line["src"],
            "mt": line["tgt"][:int(p*fertility_train[line["langs"]]*len(line["src"]))],
            "score": line["score"],
        }
        for line in data_orig
    ]
    for p in [0.25, 0.50, 0.75, 1.00]
}

for model_name, model in [
    # ("helium", model_orig),
    # ("boron", model_boron),
    ("carbon", model_carbon),
]:
    for p in [0.25, 0.50, 0.75, 1.00]:
        data = data_all[p]
        data_m_d = model.predict(data, batch_size=32)["scores"]
        print(
            f"Correlation model={model_name} data={p:.0%}",
            f'{scipy.stats.pearsonr(data_m_d, [float(x["score"]) for x in data]).correlation:.3f}',
        )

"""
sbatch_gpu_short "eval_partial" "python3 ../COMET-early-exit-experiments/vilem/09-eval_partial_comet.py"
"""