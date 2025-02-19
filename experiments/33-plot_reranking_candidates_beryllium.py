# %%

import json
import numpy as np
import random
import matplotlib.pyplot as plt
import utils_figs

r_choice = random.Random(0)

FNAME = "beam"
FNAME = "sample"

data = json.load(open(f"../computed/score_candidates_reranking_{FNAME}.json", "r"))

def perform_recipe_earlyexit(data, layer_i):
    final_scores = []
    final_top1s = []
    for line in data:
        # take max based on layer_i
        final_score = max(line["scores"], key=lambda x: x[layer_i])
        final_score = final_score[-1]

        # did we select the overall best one?
        final_top1 = max([x[-1] for x in line["scores"]]) == final_score

        final_scores.append(final_score)
        final_top1s.append(final_top1)

    return np.average(final_top1s), np.average(final_scores)

def perform_recipe_random(data, budget):
    final_scores = []
    final_top1s = []
    for line in data:
        # what's the final score?
        final_score = max(r_choice.sample(line["scores"], k=budget), key=lambda x: x[-1])[-1]
        # did we select the overall best one?
        final_top1 = max([x[-1] for x in line["scores"]]) == final_score

        final_scores.append(final_score)
        final_top1s.append(final_top1)

    return np.average(final_top1s), np.average(final_scores)

data_plot_x = []
data_plot_helium_top1 = []
data_plot_beryllium_top1 = []
data_plot_random_top1 = []
data_plot_helium_topS = []
data_plot_beryllium_topS = []
data_plot_random_topS = []
for layer_i in range(1, 24+1):
    cost = layer_i / 24
    out_helium = perform_recipe_earlyexit(data["helium"], layer_i)
    data_plot_helium_top1.append((cost, out_helium[0]))
    data_plot_helium_topS.append((cost, out_helium[1]))

    out_beryllium = perform_recipe_earlyexit(data["beryllium"], layer_i)
    data_plot_beryllium_top1.append((cost, out_beryllium[0]))
    data_plot_beryllium_topS.append((cost, out_beryllium[1]))

n_candidates = min([len(line["scores"]) for line in data["beryllium"]])
for budget in range(1, n_candidates+1):
    cost = budget / (n_candidates+1)
    out_random = perform_recipe_random(data["beryllium"], budget)
    data_plot_random_top1.append((cost, out_random[0]))
    data_plot_random_topS.append((cost, out_random[1]))

"""
scp euler:/cluster/work/sachan/vilem/COMET-early-exit-experiments/logs/score_candidates_reranking_beam.out computed/score_candidates_reranking_beam.json
scp euler:/cluster/work/sachan/vilem/COMET-early-exit-experiments/logs/score_candidates_reranking_sample.out computed/score_candidates_reranking_sample.json
"""

# sort by cost
# data_plot_helium_top1 = sorted(data_plot_helium_top1, key=lambda x: x[0])
# data_plot_beryllium_top1 = sorted(data_plot_beryllium_top1, key=lambda x: x[0])

plt.figure(figsize=(4, 2))
plt.plot(
    [x[0] for x in data_plot_beryllium_top1],
    np.array([x[1] for x in data_plot_beryllium_top1]),
    label="Partial COMET",
    color=utils_figs.COLORS[0],
)
plt.plot(
    [x[0] for x in data_plot_random_top1],
    np.array([x[1] for x in data_plot_random_top1]),
    label="Random",
    color=utils_figs.COLORS[2],
)
plt.plot(
    [x[0] for x in data_plot_helium_top1],
    np.array([x[1] for x in data_plot_helium_top1]),
    label="Full COMET",
    color=utils_figs.COLORS[1],
)

plt.legend(handletextpad=0.1)
plt.ylabel("Final candidate is top-1")
plt.xlabel("Cost")
plt.xticks([0.3, 0.5, 0.75, 1.0], ["30%", "50%", "75%", "100%"])
utils_figs.turn_off_spines()
plt.tight_layout(pad=0.5)
# plt.savefig(f"../figures/31-partial_candidates_top1_{FNAME}.pdf")


import matplotlib.pyplot as plt
import utils_figs

# sort by cost
# data_plot_helium_topS = sorted(data_plot_helium_topS, key=lambda x: x[0])
# data_plot_beryllium_topS = sorted(data_plot_beryllium_topS, key=lambda x: x[0])

plt.figure(figsize=(4, 2))
plt.plot(
    [x[0] for x in data_plot_beryllium_topS],
    np.array([x[1] for x in data_plot_beryllium_topS]) / max([x[1] for x in data_plot_beryllium_topS]),
    label="Partial COMET",
    color=utils_figs.COLORS[0],
)
plt.plot(
    [x[0] for x in data_plot_random_topS],
    np.array([x[1] for x in data_plot_random_topS]) / max([x[1] for x in data_plot_random_topS]),
    label="Random",
    color=utils_figs.COLORS[2],
)
plt.plot(
    [x[0] for x in data_plot_helium_topS],
    np.array([x[1] for x in data_plot_helium_topS]) / max([x[1] for x in data_plot_helium_topS]),
    label="Full COMET",
    color=utils_figs.COLORS[1],
)

plt.ylim(0.96, None)
plt.legend(handletextpad=0.1)
plt.ylabel("Final candidate score")
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.xticks([0.3, 0.5, 0.75, 1.0], ["30%", "50%", "75%", "100%"])

plt.xlabel("Cost")
utils_figs.turn_off_spines()
plt.tight_layout(pad=0.5)
# plt.savefig(f"../figures/31-partial_candidates_topS_{FNAME}.pdf")
