import comet
import scipy.stats
import json
from utils import FERTILITY

data_orig = [json.loads(line) for line in open("data/jsonl/test.jsonl")]

model_helium = comet.load_from_checkpoint("lightning_logs/version_22504094/checkpoints/epoch=4-step=29320-val_pearson=0.419.ckpt")
model_carbon = comet.load_from_checkpoint("lightning_logs/version_22525421/checkpoints/epoch=4-step=29320-val_pearson=0.380.ckpt")

data_all = {
    p: [
        {
            "src": line["src"],
            "mt": line["tgt"][:int(p*FERTILITY[line["langs"]]*len(line["src"]))],
            "score": line["score"],
        }
        for line in data_orig
    ]
    for p in [0.25, 0.50, 0.75, 10.00]
}

# NOTE: use 1_000% (10x) to make sure the whole translation is in

for model_name, model in [
    ("helium", model_helium),
    ("carbon", model_carbon),
]:
    for p in [0.25, 0.50, 0.75, 10.00]:
        data = data_all[p]
        data_m_d = model.predict(data, batch_size=64)["scores"]
        print(
            f"Correlation model={model_name} data={p:.0%}",
            f'{scipy.stats.pearsonr(data_m_d, [float(x["score"]) for x in data]).correlation:.3f}',
        )

"""
sbatch_gpu_big_short "eval_partial" "python3 ../COMET-early-exit-experiments/vilem/09-eval_partial_comet.py"
sbatch_gpu_big_short "eval_partial_carbon" "python3 ../COMET-early-exit-experiments/vilem/09-eval_partial_comet.py"
sbatch_gpu_big_short "eval_partial_10x" "python3 ../COMET-early-exit-experiments/vilem/09-eval_partial_comet.py"
"""