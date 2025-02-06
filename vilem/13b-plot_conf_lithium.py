# %%

import json
import utils_figs
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

data = json.load(open("../computed/eval_lithium_conf.json", "r"))

# %%

arr_conf = np.array([x["confidence"] for x in data])
arr_score = np.array([x["scores"] for x in data])
arr_human = np.array([x["human"] for x in data])

# duplicate [A] to [A, A, A]
arr_human_m = np.tile(arr_human, (arr_score.shape[1], 1)).T

plt.figure(figsize=(3, 2))
plt.plot(
    np.average(np.abs(arr_score-arr_human_m), axis=0),
    label="True error",
    color="black",
)
plt.plot(
    np.average(arr_conf, axis=0),
    label="Confidence",
    color="gray",
)
plt.legend(
    handlelength=1,
    handletextpad=0.5,
)
plt.ylabel("Mean Absolute Error")

# compute correlation at each layer

xticks = []
for i in range(arr_conf.shape[1]):
    # err_true = np.abs(arr_score[:,i]-arr_human)
    # the error is to the last layer, not human!
    err_true = np.abs(arr_score[:,i]-arr_human)
    print(
        f"Layer {i:0>2}:",
        f"score_corr={scipy.stats.pearsonr(arr_score[:,i], arr_human).correlation:.2f}",
        f"conf_corr={scipy.stats.pearsonr(arr_conf[:,i], err_true).correlation:.2f}"
    )
    xticks.append(f"Layer {i}\n({scipy.stats.pearsonr(arr_conf[:,i], err_true).correlation:.2f})")

plt.xticks(
    range(3, len(xticks), 6),
    xticks[3::6],
    fontsize=8,
)
utils_figs.turn_off_spines()
plt.tight_layout(pad=0)
plt.xlabel("Layer (correlation)")
plt.savefig("../figures/13-plot_conf.pdf")