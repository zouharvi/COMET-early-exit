# %%

import json
from typing import Tuple
import utils_figs
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

data = json.load(open("../computed/eval_beryllium_conf.json", "r"))

# %%

# train linear transformation to predict human score from layer scores
layer_to_human = []
for i in range(25):
    out_linregress = scipy.stats.linregress(
        [x["scores"][i] for x in data],
        # [x["scores"][-1] for x in data]
        [x["human"] for x in data]
    )
    layer_to_human.append((out_linregress.intercept, out_linregress.slope))
# %%

# baseline is just each layer

plot_x_base = []
plot_y_base = []
for layer in range(25):
    plot_x_base.append(layer/25)
    plot_y_base.append(
        scipy.stats.pearsonr(
            [x["scores"][layer] for x in data],
            # [x["scores"][-1] for x in data]
            [x["human"] for x in data]
        ).correlation
    )
def run_earlyexit1(line, threshold) -> Tuple[float, float]:
    """
    returns (score, layer)
    """
    for i in range(25):
        if line["confidence"][i] < threshold:
            return line["scores"][i], i
    
    return line["scores"][-1], 24

def align(score, layer):
    return (
        layer_to_human[layer][0] + layer_to_human[layer][1]*score,
        layer/25
    )

plot_x_ee1 = []
plot_y_ee1 = []
for threshold in [0.1, 0.5, 1, 2, 3, 4, 5, 6, 8, 9, 9.01, 9.02, 9.03, 9.04, 9.05, 10]:
    out = [run_earlyexit1(line, threshold) for line in data]
    out = [align(score, layer) for score, layer in out]
    plot_x_ee1.append(np.average([x[1] for x in out]))
    plot_y_ee1.append(
        scipy.stats.pearsonr(
            [x[0] for x in out],
            # [x["scores"][-1] for x in data]
            [x["human"] for x in data]
        ).correlation
    )

# %%


def run_earlyexit2(line, threshold) -> Tuple[float, float]:
    """
    returns (score, layer)
    """
    for i in range(25):
        if np.var(line["scores"][i-2:i+1]) < threshold:
            return np.average(line["scores"][i-2:i+1]), i
    
    return line["scores"][-1], 24

def align(score, layer):
    return (
        layer_to_human[layer][0] + layer_to_human[layer][1]*score,
        layer/25
    )

plot_x_ee2 = []
plot_y_ee2 = []
for threshold in [0, 0.001, 0.005, 0.1, 0.5, 1, 10]:
    out = [run_earlyexit2(line, threshold) for line in data]
    out = [align(score, layer) for score, layer in out]
    plot_x_ee2.append(np.average([x[1] for x in out]))
    plot_y_ee2.append(
        scipy.stats.pearsonr(
            [x[0] for x in out],
            # [x["scores"][-1] for x in data]
            [x["human"] for x in data]
        ).correlation
    )


# %%

plt.figure(figsize=(3, 2))
plt.plot(
    plot_x_base, plot_y_base,
    label="Constant-Exit",
)
plt.plot(
    plot_x_ee1, plot_y_ee1,
    label="Threshold-Exit",
)
plt.plot(
    plot_x_ee2, plot_y_ee2,
    label="Variance-Exit",
)
plt.legend()
utils_figs.turn_off_spines()
plt.ylabel("Correlation")
plt.xlabel("Cost")
# percentage xticks
plt.xticks(
    [0, 0.25, 0.5, 0.75, 1],
    ["0%", "25%", "50%", "75%", "100%"]
)
plt.tight_layout(pad=0)
plt.savefig("../figures/20-early_exit_alg.pdf")
plt.show()