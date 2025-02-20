import comet_early_exit
import csv
import argparse

args = argparse.ArgumentParser()
args.add_argument("data_in")
args.add_argument("data_out")
args = args.parse_args()

model = comet_early_exit.load_from_checkpoint("lightning_logs/version_22504094/checkpoints/epoch=4-step=29320-val_pearson=0.419.ckpt")
data = list(csv.DictReader(open(args.data_in, "r")))
out = model.predict(data, batch_size=128)

data_out = []
for pred, line in zip(out["scores"], data):
    data_out.append({
        "src": line["src"],
        "mt": line["mt"],
        "score": abs(float(line["score"]) - pred),
        "predicted": pred,
    })


with open(args.data_out, "w") as f:
    writer = csv.DictWriter(f, fieldnames=["src", "mt", "score", "predicted"])
    writer.writeheader()
    writer.writerows(data_out)

"""
sbatch_gpu_big_short "generate_test_sodium" "python3 ../COMET-early-exit-experiments/vilem/50-generate_sodium.py data/csv/test_da.csv data/csv/test_sodium.csv"
sbatch_gpu_big_short "generate_train_sodium" "python3 ../COMET-early-exit-experiments/vilem/50-generate_sodium.py data/csv/train_da.csv data/csv/train_sodium.csv"
sbatch_gpu_big_short "generate_dev_sodium" "python3 ../COMET-early-exit-experiments/vilem/50-generate_sodium.py data/csv/dev_da.csv data/csv/dev_sodium.csv"
"""