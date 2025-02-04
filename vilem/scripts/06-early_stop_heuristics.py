# %%

import comet
import csv
import scipy.stats
import numpy as np
import random
import matplotlib.pyplot as plt
import os
os.chdir("/home/vilda/comet-early-exit")

model = comet.load_from_checkpoint("lightning_logs/hydrogen/checkpoints/model.ckpt")
data = random.Random(0).sample(list(csv.DictReader(open("data/csv/dev_da.csv", "r"))), k=1_000)
pred_y = np.array(model.predict(data, batch_size=32)["scores"])
data_y = [float(x["score"]) for x in data]

# %%
def heuristics_confidence_interval(preds, threshold):
    val_lower, val_upper = scipy.stats.t.interval(0.95, len(preds) - 1, loc=np.mean(preds), scale=scipy.stats.sem(preds))
    return val_upper - val_lower < threshold

def heuristics_std(preds, threshold):
    print(np.std(preds))
    return np.std(preds) < threshold

def first_stable(preds, threshold):
    for i in range(3, len(preds)):
        if heuristics_confidence_interval(preds[i-3:i+1], threshold):
            return (i, preds[i])
    return (len(preds)-1, preds[-1])


earylexit_computation = []
earylexit_corr = []
for threshold in [0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8]:
    pred_y_heuristics = []
    layer_heuristics = []
    for pred_y_line in pred_y:
        i, pred = first_stable(pred_y_line, threshold)
        pred_y_heuristics.append(pred)
        layer_heuristics.append(i)
    
    corr = scipy.stats.pearsonr(data_y, pred_y_heuristics).correlation
    earylexit_computation.append((np.average(layer_heuristics)+1)/len(pred_y[0]))
    earylexit_corr.append(corr)

# %%
base_computation = []
base_corr = []
for i, preds in enumerate(pred_y.transpose(1, 0)):
    corr = scipy.stats.pearsonr(data_y, preds).correlation
    base_computation.append((i+1)/len(pred_y[0]))
    base_corr.append(corr)

plt.plot(base_computation, base_corr, label="fixed early exit")
plt.plot(earylexit_computation, earylexit_corr, label="early exit with self-confidence")
plt.xlabel("Computation")
plt.ylabel("Correlation with humans")
plt.legend()
