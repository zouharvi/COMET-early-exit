import comet
import csv
import scipy.stats

data_part = list(csv.DictReader(open('data/csv/dev_partial.csv')))
data_orig = list(csv.DictReader(open('data/csv/dev_da.csv')))

model_part = comet.load_from_checkpoint("lightning_logs/version_22525419/checkpoints/epoch=4-step=29320-val_pearson=0.377.ckpt")
model_orig = comet.load_from_checkpoint("lightning_logs/version_22504094/checkpoints/epoch=4-step=29320-val_pearson=0.419.ckpt")

data_morig_dorig = model_orig.predict(data_orig, batch_size=32)["scores"]
data_morig_dpart = model_orig.predict(data_part, batch_size=32)["scores"]
data_mpart_dorig = model_part.predict(data_orig, batch_size=32)["scores"]
data_mpart_dpart = model_part.predict(data_part, batch_size=32)["scores"]


print(
    "Correlation model=orig data=orig",
    scipy.stats.pearsonr(data_morig_dorig, [float(x["score"]) for x in data_orig]).correlation
)
print(
    "Correlation model=orig data=part",
    scipy.stats.pearsonr(data_morig_dpart, [float(x["score"]) for x in data_part]).correlation
)
print(
    "Correlation model=part data=orig",
    scipy.stats.pearsonr(data_mpart_dorig, [float(x["score"]) for x in data_orig]).correlation
)
print(
    "Correlation model=part data=part",
    scipy.stats.pearsonr(data_mpart_dpart, [float(x["score"]) for x in data_part]).correlation
)

"""
sbatch_gpu_short "eval_partial" "python3 ../COMET-early-exit-experiments/vilem/09-eval_partial_comet.py"
"""