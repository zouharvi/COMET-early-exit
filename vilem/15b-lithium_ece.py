# %%

import json

import sklearn.linear_model
import utils_figs
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

# TODO: use all data
data = json.load(open("../computed/eval_lithium_conf.json", "r"))

# %%
arr_score = np.array([x["scores"][-1] for x in data])
arr_human = np.array([x["human"] for x in data])
arr_conf = np.array([x["confidence"][-1] for x in data])

hum_corr = scipy.stats.pearsonr(
    arr_score, arr_human,
).correlation
error_corr = scipy.stats.pearsonr(
    arr_conf,
    np.abs(arr_score-arr_human),
).correlation
print(f"{hum_corr:.3f} {error_corr:.3f}")

# %%
# ECE

# 100 bins based on arr_conf such that each bin has same number of samples
bins = np.array_split(-np.sort(-arr_conf), 100)
bins = [
    max(bin) for bin in bins
] + [min(bins[-1])]

bin_i = np.digitize(arr_conf, bins)

plot_x = []
plot_y = []
for i in range(0, len(bins)):
    mask = (bin_i == i)
    arr_human_local = arr_human[mask]
    arr_score_local = arr_score[mask]
    mae = np.mean(np.abs(arr_score_local - arr_human_local))
    plot_x.append(i)
    plot_y.append(mae)

plt.figure(figsize=(3, 2))

model = sklearn.linear_model.LinearRegression(fit_intercept=False)
model.fit(
    -(np.array(plot_x).reshape(-1, 1)-100),
    np.array(plot_y)
)

plt.plot(
    [0, len(bins)],
    [model.coef_[0]*100, min(plot_y)],
    color="black",
)
plt.plot(plot_x, plot_y)
utils_figs.turn_off_spines()
plt.ylabel("Prediction error")
plt.xlabel("Confidence bins")
plt.tight_layout(pad=0)
plt.savefig("../figures/15b-lithium_ece.pdf")