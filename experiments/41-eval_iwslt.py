from datasets import load_dataset
import argparse
import scipy.stats
import comet_early_exit

args = argparse.ArgumentParser()
args.add_argument("model", type=str)
args = args.parse_args()

data = load_dataset("IWSLT/da2023")["train"]
data = [
    {
        "src": x["src"],
        "mt": x["mt"],
        "score": x["raw"],
    }
    for x in data
]

model = comet_early_exit.load_from_checkpoint(args.model)
scores_pred = model.predict(data, batch_size=128)["scores"]

print(scipy.stats.pearsonr([x["score"] for x in data], scores_pred))

"""
sbatch_gpu_big_short "eval_iwslt_helium" "python3 ../COMET-early-exit-experiments/vilem/41-eval_iwslt.py lightning_logs/version_22504094/checkpoints/epoch=4-step=29320-val_pearson=0.419.ckpt"
sbatch_gpu_big_short "eval_iwslt_fluorine" "python3 ../COMET-early-exit-experiments/vilem/41-eval_iwslt.py lightning_logs/version_22525436/checkpoints/epoch=4-step=29320-val_pearson=0.378.ckpt"
sbatch_gpu_big_short "eval_iwslt_carbon" "python3 ../COMET-early-exit-experiments/vilem/41-eval_iwslt.py lightning_logs/version_22525421/checkpoints/epoch=4-step=29320-val_pearson=0.380.ckpt"
sbatch_gpu_big_short "eval_iwslt_neon" "python3 ../COMET-early-exit-experiments/vilem/41-eval_iwslt.py lightning_logs/version_22525436/checkpoints/epoch=4-step=29320-val_pearson=0.378.ckpt"
"""