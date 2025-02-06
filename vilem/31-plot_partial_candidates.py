# %%

import json
import itertools
import numpy as np

FNAME = "beam"
# FNAME = "sample"

data = json.load(open(f"../computed/score_candidates_partial_{FNAME}.json", "r"))

def perform_recipe(data, recipe):
    final_scores = []
    final_top1s = []
    for line in data:
        # we start with all candidates
        candidates = line["scores"]
        for p, recipe_p in zip([0.25, 0.50, 0.75], recipe):
            # remove bottom recipe_p canddidates from candidates[p]
            candidates_sorted = sorted(candidates, key=lambda x: x[str(p)], reverse=True)
            candidates = candidates_sorted[:int(recipe_p * len(candidates_sorted))]

        # what's the final score?
        final_score = max([x["10.0"] for x in candidates])
        # did we select the overall best one?
        final_top1 = max([x["10.0"] for x in line["scores"]]) == final_score

        final_scores.append(final_score)
        final_top1s.append(final_top1)

    return np.average(final_scores), np.average(final_top1s)

# cut at 0.25, 0.5, 0.75
RECIPES = itertools.product([0.25, 0.5, 0.75, 1.0], repeat=3)

data_plot_x = []
data_plot_helium_top1 = []
data_plot_carbon_top1 = []
data_plot_helium_topS = []
data_plot_carbon_topS = []
for recipe in RECIPES:
    cost = (
        0.25 +
        0.25 * recipe[0] +
        0.25 * recipe[0] * recipe[1] +
        0.25 * recipe[0] * recipe[1] * recipe[2]
    )
    data_plot_x.append(cost)
    # if cost < 0.41:
    #     continue

    out_helium = perform_recipe(data["helium"], recipe)
    data_plot_helium_top1.append((cost, out_helium[0]))
    data_plot_helium_topS.append((cost, out_helium[1]))

    out_carbon = perform_recipe(data["carbon"], recipe)
    data_plot_carbon_top1.append((cost, out_carbon[0]))
    data_plot_carbon_topS.append((cost, out_carbon[1]))

"""
scp euler:/cluster/work/sachan/vilem/COMET-early-exit-experiments/logs/score_candidates_partial_beam.out computed/score_candidates_partial_beam.json
scp euler:/cluster/work/sachan/vilem/COMET-early-exit-experiments/logs/score_candidates_partial_sample.out computed/score_candidates_partial_sample.json
"""

import matplotlib.pyplot as plt
import utils_figs

# sort by cost
data_plot_helium_top1 = sorted(data_plot_helium_top1, key=lambda x: x[0])
data_plot_carbon_top1 = sorted(data_plot_carbon_top1, key=lambda x: x[0])

plt.figure(figsize=(4, 2))
plt.scatter(
    [x[0] for x in data_plot_carbon_top1],
    np.array([x[1] for x in data_plot_carbon_top1]) / max([x[1] for x in data_plot_carbon_top1]),
    label="Partial COMET",
    marker=".",
)
plt.scatter(
    [x[0] for x in data_plot_helium_top1],
    np.array([x[1] for x in data_plot_helium_top1]) / max([x[1] for x in data_plot_helium_top1]),
    label="Full COMET",
    marker=".",
)

plt.tight_layout(pad=0)
plt.legend(handletextpad=0.1)
plt.ylabel("Final candidate score")
plt.xlabel("Generative model cost")
plt.xticks([0.3, 0.5, 0.75, 1.0], ["0.30", "0.5", "0.75", "1.0"])
utils_figs.turn_off_spines()
plt.savefig("../figures/31-partial_candidates_top1.pdf")




import matplotlib.pyplot as plt
import utils_figs

# sort by cost
data_plot_helium_topS = sorted(data_plot_helium_topS, key=lambda x: x[0])
data_plot_carbon_topS = sorted(data_plot_carbon_topS, key=lambda x: x[0])

plt.figure(figsize=(4, 2))
plt.scatter(
    [x[0] for x in data_plot_carbon_topS],
    np.array([x[1] for x in data_plot_carbon_topS]) / max([x[1] for x in data_plot_carbon_topS]),
    label="Partial COMET",
    marker=".",
)
plt.scatter(
    [x[0] for x in data_plot_helium_topS],
    np.array([x[1] for x in data_plot_helium_topS]) / max([x[1] for x in data_plot_helium_topS]),
    label="Full COMET",
    marker=".",
)

plt.tight_layout(pad=0)
plt.legend(handletextpad=0.1)
plt.ylabel("Final candidate is top-1")
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.xticks([0.3, 0.5, 0.75, 1.0], ["0.30", "0.5", "0.75", "1.0"])

plt.xlabel("Generative model cost")
utils_figs.turn_off_spines()
plt.savefig("../figures/31-partial_candidates_topS.pdf")
