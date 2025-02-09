import comet
import csv
import scipy.stats

model = comet.load_from_checkpoint("lightning_logs/version_22525438/checkpoints/epoch=4-step=29320-val_pearson=0.387.ckpt")
data = list(csv.DictReader(open("data/csv/test_sodium.csv", "r")))
pred_error = model.predict(data, batch_size=128)["scores"]

print(scipy.stats.pearsonr([float(x["score"]) for x in data], pred_error))

"""
sbatch_gpu_big_short "eval_sodium" "python3 ../COMET-early-exit-experiments/vilem/51-eval_sodium.py"
"""