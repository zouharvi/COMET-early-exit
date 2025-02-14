# %%

import matplotlib.pyplot as plt
import utils_figs
import pickle
import h5py

with open("../computed/bandit_start_14324_ucb_10_normbeam.pkl", "rb") as f:
    data_bandit = pickle.load(f)

data_bayes = h5py.File("../computed/gp_all_0.25_10_sample.h5", "r")

# process bayes data
for k in ["bayesopt", "random"]:
    data_x = []
    data_y = []
    for i, x in enumerate(data_bayes[k+ "_best_retrieved"]):
        if x != 0.0:
            data_x.append(i/200)
            data_y.append(x)

    data_bandit["scores"][k] = {}
    data_bandit["scores"][k]["win_rate"] = (data_x, data_y)

    data_x = []
    data_y = []
    for i, x in enumerate(data_bayes[k+"_score"]):
        if x != 0.0:
            data_x.append(i/200)
            data_y.append(x)
    data_bandit["scores"][k]["avg_score"] = (data_x, data_y)

METHOD_TO_NAME = {
    'bandit (0, 1.0)': "Bandit ($\\gamma=1.0$)",
    'random baseline': "Random",
    'top-k sum logprob': "LogProb Sum",
    # 'top-k avg logprob': "LogProb Avg",
    'bayesopt': "BayesOpt",
    "random": "Random from Bayes"
}
name = "sample"

# %%

for mode in ["avg_score", "win_rate"]:
    plt.figure(figsize=(4, 2))
    for method in METHOD_TO_NAME:
        data_local = zip(*data_bandit["scores"][method][mode])
        data_local = [x for x in data_local if x[0] > 0.1]
        plt.plot(
            [x[0] for x in data_local],
            [x[1] for x in data_local],
            label=METHOD_TO_NAME[method],
        )
        
    plt.xticks(
        [0.2, 0.4, 0.6, 0.8, 1.0],
        ["20%", "40%", "60%", "80%", "100%"],
    )
    plt.xlabel("Cost")
    plt.ylabel("Final candidate score" if mode == "avg_score" else "Final candidate is top-1    ")

    if mode == "win_rate":
        plt.gca().set_yticklabels(["{:,.0%}".format(x) for x in plt.gca().get_yticks()])
    elif mode == "avg_score":
        plt.gca().set_yticklabels(["{:,.1f}".format(x) for x in plt.gca().get_yticks()])

    if mode == "win_rate":
        plt.legend(
            labelspacing=0.2,
        )
    utils_figs.turn_off_spines()
    plt.tight_layout(pad=0.1)
    plt.savefig(f"../figures/55-bandit_vs_bayes_{mode}_{name}.pdf")
