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

def plot_layer_average(key):
    plt.plot(
        range(24),
        [np.average([x["scores"][i] for x in data[key]]) for i in range(24)],
        label=key,
    )

plot_layer_average("helium")
plot_layer_average("beryllium")
plt.xlabel("Layer")
plt.ylabel("Average score")
plt.legend()
plt.show()