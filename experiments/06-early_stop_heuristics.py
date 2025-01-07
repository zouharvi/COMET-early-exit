# %%

import comet
import csv
import scipy.stats
import numpy as np
import random
import os
os.chdir("/home/vilda/comet-early-exit")

model = comet.load_from_checkpoint("lightning_logs/version_19245793/checkpoints/epoch=3-step=23456-val_avg_pearson=0.257.ckpt")
data = random.Random(0).sample(list(csv.DictReader(open("data/csv/dev_da.csv", "r"))), k=1_000)
pred_y = np.array(model.predict(data, batch_size=32)["scores"])
data_y = [float(x["score"]) for x in data]

# %%
import matplotlib.pyplot as plt

plt.plot(pred_y[0])
plt.xlabel("Layer")
plt.ylabel("Prediction")

# %%
def heuristics_confidence_interval(preds):
    val_lower, val_upper = scipy.stats.t.interval(0.95, len(preds) - 1, loc=np.mean(preds), scale=scipy.stats.sem(preds))
    return val_upper - val_lower < 2

def first_stable(preds):
    for i in range(3, len(preds)):
        if heuristics_confidence_interval(preds[i-3:i+1]):
            return (i, preds[i])
    return (len(preds)-1, preds[-1])

pred_y_heuristics = []
layer_heuristics = []
for pred_y_line in pred_y:
    i, pred = first_stable(pred_y_line)
    pred_y_heuristics.append(pred)
    layer_heuristics.append(i)
corr = scipy.stats.pearsonr(data_y, pred_y_heuristics).correlation
print(f"Computation {(np.average(layer_heuristics)+1)/len(pred_y[0]):.1%}: corr={corr:.1%}")

# %%
for i, preds in enumerate(pred_y.T):
    corr = scipy.stats.pearsonr(data_y, preds).correlation
    print(f"Computation {(i+1)/len(pred_y[0]):.1%}: corr={corr:.1%}")