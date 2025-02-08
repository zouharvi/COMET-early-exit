# %%

import json
import numpy as np
import random
import matplotlib.pyplot as plt
import tqdm
import utils_figs
import matplotlib.pyplot as plt
import utils_figs
from multiprocessing.pool import ThreadPool

r_choice = random.Random(0)

FNAME = "beam"
# FNAME = "sample"

data = json.load(open(f"../computed/score_candidates_reranking_{FNAME}.json", "r"))
n_candidates = min([len(line["scores"]) for line in data["beryllium"]])

# TODO: don't use confidence interval?

def perform_recipe_bandit(data):
    def perform_bandit_local(line):
        start_layer = 15
        # we have to pay for the first layer
        cost = n_candidates / 24 * (start_layer+1)
        candidates = list(zip(range(len(line["scores"])), line["scores"], line["confidences"]))
        candidates = [[x[0], x[1][start_layer:], x[2][start_layer:]] for x in candidates]

        final_costs = []
        final_scores = []
        final_top1s = []

        # take max based on layer_i
        while True:
            cost += 1 / 24
            # reveal one with the highest upper bound (score + confidence)
            # that can be more revealed
            candidates_to_reveal = [x for x in candidates if len(x[1]) >= 2]
            if len(candidates_to_reveal) == 0:
                break
            champion = max(candidates_to_reveal, key=lambda x: x[1] + x[2])
            # reveal next layer for both score and confidence
            candidates[champion[0]][1] = candidates[champion[0]][1][1:]
            candidates[champion[0]][2] = candidates[champion[0]][2][1:]

            # final score based on the last layer
            final_score = max(candidates, key=lambda x: x[1][0])[1][-1]

            # did we select the overall best one?
            final_top1 = max([x[-1] for x in line["scores"]]) == final_score

            final_costs.append(cost)
            final_scores.append(final_score)
            final_top1s.append(final_top1)

        return final_top1s, final_scores, final_costs
        # final_scores.append(final_score)
        # final_top1s.append(final_top1)

    with ThreadPool(16) as pool:
        out = pool.map(perform_bandit_local, tqdm.tqdm(data))
        out = np.array(out)
        out = np.average(out, axis=0)

    return out[0], out[1], out[2]

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


data_plot_beryllium_top1, data_plot_beryllium_topS, data_plot_beryllium_cost = perform_recipe_bandit(data["beryllium"])

data_plot_random_top1 = []
data_plot_random_topS = []
for budget in tqdm.tqdm(range(1, n_candidates+1)):
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
    [x/n_candidates for x in data_plot_beryllium_cost],
    data_plot_beryllium_top1,
    label="Bandit",
    color=utils_figs.COLORS[0],
)
plt.plot(
    [x[0] for x in data_plot_random_top1],
    np.array([x[1] for x in data_plot_random_top1]),
    label="Random",
    color=utils_figs.COLORS[2],
)

plt.legend(handletextpad=0.1)
plt.ylabel("Final candidate is top-1")
plt.xlabel("Cost")
plt.xticks([0.3, 0.5, 0.75, 1.0], ["30%", "50%", "75%", "100%"])
utils_figs.turn_off_spines()
plt.tight_layout(pad=0.5)
# plt.savefig(f"../figures/31-partial_candidates_top1_{FNAME}.pdf")



# sort by cost
# data_plot_helium_topS = sorted(data_plot_helium_topS, key=lambda x: x[0])
# data_plot_beryllium_topS = sorted(data_plot_beryllium_topS, key=lambda x: x[0])

plt.figure(figsize=(4, 2))
plt.plot(
    [x/n_candidates for x in data_plot_beryllium_cost],
    np.array(data_plot_beryllium_topS) / max(data_plot_beryllium_topS),
    label="Bandit",
    color=utils_figs.COLORS[0],
)
plt.plot(
    [x[0] for x in data_plot_random_topS],
    np.array([x[1] for x in data_plot_random_topS]) / max([x[1] for x in data_plot_random_topS]),
    label="Random",
    color=utils_figs.COLORS[2],
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
