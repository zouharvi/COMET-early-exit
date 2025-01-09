# %%

import comet
import csv
import scipy.stats
import numpy as np
import random
import matplotlib.pyplot as plt
import os
os.chdir("/home/vilda/comet-early-exit")

model = comet.load_from_checkpoint("lightning_logs/beryllium/checkpoints/model.ckpt")
data = random.Random(0).sample(list(csv.DictReader(open("data/csv/dev_da.csv", "r"))), k=1_000)
output = model.predict(data, batch_size=32)
pred_y = np.stack((output["scores"], output["confidences"]), axis=-1)
data_y = [float(x["score"]) for x in data]

# %%

plt.plot(pred_y[0,:,0], label="score")
plt.fill_between(
    range(len(pred_y[0,:,0])),
    pred_y[0,:,0] - pred_y[0,:,1],
    pred_y[0,:,0] + pred_y[0,:,1],
    alpha=0.5
)
plt.legend()
plt.xlabel("Layer")
plt.ylabel("Prediction")

# %%
def heuristics_self_confidence(preds, threshold):
    confidences = preds[:,1]
    return confidences[-1] < threshold

def first_stable(preds, threshold=3):
    for i in range(1, len(preds)):
        if heuristics_self_confidence(preds[:i+1], threshold):
            return (i, preds[i][0])
    return (len(preds)-1, preds[-1][0])

earylexit2_computation = []
earylexit2_corr = []
for threshold in [0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    pred_y_heuristics = []
    layer_heuristics = []
    for pred_y_line in pred_y:
        i, pred = first_stable(pred_y_line, threshold=threshold)
        pred_y_heuristics.append(pred)
        layer_heuristics.append(i)
    corr = scipy.stats.pearsonr(data_y, pred_y_heuristics).correlation
    earylexit2_computation.append((np.average(layer_heuristics)+1)/len(pred_y[0]))
    earylexit2_corr.append(corr)

# %%

base_computation = []
base_corr = []
for i, preds in enumerate(pred_y.transpose(1, 0, 2)):
    corr = scipy.stats.pearsonr(data_y, preds[:,0]).correlation
    base_computation.append((i+1)/len(pred_y[0]))
    base_corr.append(corr)

plt.plot(base_computation, base_corr, label="fixed early exit")
plt.plot(earylexit2_computation, earylexit2_corr, label="early exit with self-confidence")
plt.xlabel("Computation")
plt.ylabel("Correlation with humans")
plt.legend()


# %%
data_y_last = pred_y[:,-1,0]
earylexit2_computation = []
earylexit2_corr = []
for threshold in [0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    pred_y_heuristics = []
    layer_heuristics = []
    for pred_y_line in pred_y:
        i, pred = first_stable(pred_y_line, threshold=threshold)
        pred_y_heuristics.append(pred)
        layer_heuristics.append(i)
    corr = scipy.stats.pearsonr(data_y_last, pred_y_heuristics).correlation
    earylexit2_computation.append((np.average(layer_heuristics)+1)/len(pred_y[0]))
    earylexit2_corr.append(corr)

# %%

base_computation = []
base_corr = []
for i, preds in enumerate(pred_y.transpose(1, 0, 2)):
    corr = scipy.stats.pearsonr(data_y_last, preds[:,0]).correlation
    base_computation.append((i+1)/len(pred_y[0]))
    base_corr.append(corr)

# %%
plt.plot(base_computation, base_corr, label="fixed early exit")
plt.plot(earylexit2_computation, earylexit2_corr, label="early exit with self-confidence")
plt.xlabel("Computation")
plt.ylabel("Correlation with last predictdion")
plt.legend()