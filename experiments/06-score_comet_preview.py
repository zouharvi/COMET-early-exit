# %%

import comet
import csv
import scipy.stats
import numpy as np
import random
import os
os.chdir("/home/vilda/comet-early-exit")

model = comet.load_from_checkpoint("lightning_logs/version_19245790/checkpoints/epoch=3-step=46912-val_avg_pearson=0.306.ckpt")
data = random.Random(0).sample(list(csv.DictReader(open("data/csv/dev_da.csv", "r"))), k=1_000)
pred_y = np.array(model.predict(data, batch_size=32)["scores"]).T

# %%
data_y = [float(x["score"]) for x in data]
img = np.zeros((len(pred_y), len(pred_y)+1))

for layer_i, layer_y in list(enumerate(pred_y))[1:]:
    print(len(data_y), len(layer_y))
    corr_gold = scipy.stats.pearsonr(data_y, layer_y).correlation
    img[layer_i, len(pred_y)] = corr_gold
    print(
        f"Layer {layer_i:0>2}:",
        f"h={corr_gold:<5.0%}",
        end=" "
    )
    for b_layer_i, b_layer_y in list(enumerate(pred_y))[1:]:
        corr = scipy.stats.pearsonr(b_layer_y, layer_y).correlation
        img[layer_i, b_layer_i] = corr

    for b_layer_i, b_layer_y in list(enumerate(pred_y))[1::3]:
        corr = scipy.stats.pearsonr(b_layer_y, layer_y).correlation
        print(f"l{b_layer_i:0>2}={corr:<5.0%}", end=" ")
    print()

plt.imshow(img[1:,1:])
# %%
pred_y_t = pred_y.T
import matplotlib.pyplot as plt

def confidence_interval(preds):
    return scipy.stats.t.interval(0.95, len(preds) - 1, loc=np.mean(preds), scale=scipy.stats.sem(preds))

def is_stable(preds):
    return confidence_interval(preds)[1] - confidence_interval(preds)[0] < 2

# def is_stable(preds):
#     return abs(preds[-1] - preds[-2]) < 0.05

def first_stable(preds):
    for i in range(3, len(preds)):
        if is_stable(preds[i-3:i]):
            return (i, np.average(preds[i-3:i]))
    return (len(preds)-1, preds[-1])
print(first_stable(pred_y_t[0]))

print(pred_y_t[0])
plt.plot(pred_y_t[0])
plt.xlabel("Layer")
plt.ylabel("Prediction")

# %%

pred_y_heuristics = []
layer_heuristics = []
for pred_y_line in pred_y_t:
    i, pred = first_stable(pred_y_line)
    pred_y_heuristics.append(pred)
    layer_heuristics.append(i)
corr = scipy.stats.pearsonr(data_y, pred_y_heuristics).correlation
print(f"Computation {(np.average(layer_heuristics)+1)/len(pred_y_t[0]):.1%}: corr={corr:.1%}")

# %%
for i, c in enumerate(img[:,-1]):
    print(f"Computation {(i+1)/len(pred_y_t[0]):.1%}: corr={c:.1%}")