# %%

import json
import utils_figs
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

data = json.load(open("../computed/13-eval_oxygen_conf.json", "r"))


arr_conf = np.array([x["confidence"] for x in data])
arr_score = np.array([x["scores"] for x in data])
arr_human = np.array([x["human"] for x in data])

# duplicate [A] to [A, A, A]
arr_score_last_m = np.tile(arr_score[:,-1], (arr_score.shape[1], 1)).T
arr_human_m = np.tile(arr_human, (arr_score.shape[1], 1)).T

plt.figure(figsize=(4, 2))
plt.plot(
    np.average(np.abs(arr_score-arr_score_last_m), axis=0),
    # np.average(np.abs(arr_score-arr_human_m), axis=0),
    label="True error\nof each layer",
    color="black",
    linewidth=2,
)
plt.plot(
    np.average(arr_conf, axis=0),
    label="Instant Self-Confidence\nof each layer",
    color=utils_figs.COLORS[0],
    linewidth=2,
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
    err_true = np.abs(arr_score[:,i]-arr_score[:,-1])
    print(
        f"Layer {i:0>2}:",
        f"score_corr={scipy.stats.pearsonr(arr_score[:,i], arr_human).correlation:.2f}",
        f"conf_corr={scipy.stats.pearsonr(arr_conf[:,i], err_true).correlation:.2f}"
    )
    xticks.append(f"Layer {i}\n({scipy.stats.pearsonr(arr_conf[:,i], err_true).correlation:.2f})")

plt.xticks(
    range(3, len(xticks), 6),
    xticks[3::6],
    fontsize=9,
)

confidence_y = np.average(arr_conf, axis=0)
confidence_y_delta = confidence_y[1:]-confidence_y[:-1]
# top-k delta
top_k = 5
top_k_idx = np.argsort(-confidence_y_delta)[-top_k:]
print(top_k_idx+1)
utils_figs.turn_off_spines()
plt.tight_layout(pad=0)
plt.xlabel("Layer (correlation)")
plt.savefig("../figures/13-plot_conf_last.pdf")


"""
scp euler:/cluster/work/sachan/vilem/COMET-early-exit/logs/eval_nitrogen_conf.out computed/13-eval_nitrogen_conf.json
scp euler:/cluster/work/sachan/vilem/COMET-early-exit/logs/eval_oxygen_conf.out computed/13-eval_oxygen_conf.json
"""