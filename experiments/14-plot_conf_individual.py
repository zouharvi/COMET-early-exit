# %%

import json
import utils_figs
import numpy as np
import matplotlib.pyplot as plt

data = json.load(open("../computed/eval_beryllium_conf.json", "r"))

# %%

arr_conf = np.array([x["confidence"] for x in data])
arr_score = np.array([x["scores"] for x in data])
arr_human = np.array([x["human"] for x in data])

fig, axs = plt.subplots(2, 2, figsize=(3, 2))

def plot_segment(ax_i, segment_i):
    ax = axs[ax_i]
    ax.plot(
        arr_score[segment_i],
        color="#aaa",
    )

    ax.fill_between(
        range(arr_score.shape[1]),
        arr_score[segment_i]-arr_conf[segment_i],
        arr_score[segment_i]+arr_conf[segment_i],
        color="#ccc",
    )

    early_stop_i = (arr_conf[segment_i]<4).argmax()
    ax.plot(
        arr_score[segment_i][:early_stop_i],
        color="black",
    )

    ax.fill_between(
        range(early_stop_i),
        (arr_score[segment_i]-arr_conf[segment_i])[:early_stop_i],
        (arr_score[segment_i]+arr_conf[segment_i])[:early_stop_i],
        color="gray",
    )

    ax.vlines(
        x=early_stop_i-1,
        ymin=(arr_score[segment_i]-arr_conf[segment_i])[early_stop_i-1],
        ymax=(arr_score[segment_i]+arr_conf[segment_i])[early_stop_i-1],
        color=utils_figs.COLORS[0],
        linewidth=2,
    )


    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['top', 'right']].set_visible(False)
    
    if ax_i[1] == 0:
        ax.set_ylabel("Predicted")
    if ax_i[0] == 1:
        ax.set_xlabel("Layer")

plot_segment((0, 0), 0)
plot_segment((0, 1), 20)
plot_segment((1, 0), 5)
plot_segment((1, 1), 12)

# plt.xlabel("Layer (correlation)")
plt.tight_layout(pad=0.5)
plt.savefig("../figures/14-plot_conf_individual.pdf")