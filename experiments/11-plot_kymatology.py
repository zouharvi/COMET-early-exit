# %%

import csv
import matplotlib.pyplot as plt
import utils_figs
import numpy as np
import pandas as pd


def run_plot(fname, label):
    data = pd.read_csv(fname, header=None).T
    # set first row to be header
    data.columns = data.iloc[0]
    data = data[1:]
    data = data.to_dict(orient="records")

    for line in data:
        line["cost"] = np.average([v for k,v in line.items() if k.startswith("Avg. candidates left after observing score")])


    plt.plot(
        [line["cost"]/124 for line in data],
        # [line["Correct Examples:"] for line in data],
        [line["Pruned avg COMET score"]/line["Best COMET score"] for line in data],
        label=label,
    )
    # TODO

    
plt.figure(figsize=(4, 3))

run_plot("../computed/models-beryllium_multivariate_gaussians_results.csv", "Early Exit")
run_plot("../computed/models-beryllium_multivariate_gaussians_results_with_error_pred.csv", "Conf.-aware Early Exit")
run_plot("../computed/models-helium_multivariate_gaussians_results.csv", "Baseline")



utils_figs.turn_off_spines()

# xticks percentages
plt.xticks([0, 0.25, 0.5, 0.75, 1])
plt.gca().set_xticklabels(["{:,.0%}".format(x) for x in plt.gca().get_xticks()])
plt.xlabel("Cost")
plt.ylabel("Candidate score (ratio to best one)")
plt.legend()
plt.savefig(f"../figures/11-plot_kymatology.pdf")
plt.show()