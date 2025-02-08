import comet
import csv
import scipy.stats

model = comet.load_from_checkpoint("lightning_logs/version_22525438/checkpoints/epoch=0-step=5864-val_pearson=0.375.ckpt")
data = list(csv.DictReader(open("data/csv/test_sodium.csv", "r")))
out = model.predict(data, batch_size=128)["scores"]

print(scipy.stats.pearsonr([float(x["score"]) for x in data], out))

"""
sbatch_gpu_big_short "eval_sodium_e0" "python3 ../COMET-early-exit-experiments/vilem/51-eval_sodium.py"
"""