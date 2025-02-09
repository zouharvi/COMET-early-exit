# %%

import matplotlib.pyplot as plt
import utils_figs
import pickle

# TODO: plot ablation in the appendix

with open("../computed/bandit_start_14324_ucb_10_normbeam.pkl", "rb") as f:
    data_sample = pickle.load(f)
# with open("../computed/bandit_start_14324_ucb_05_normbeam.pkl", "rb") as f:
#     data_sample_05 = pickle.load(f)
with open("../computed/bandit_start_16000_ucb_10_normsample.pkl", "rb") as f:
    data_beam = pickle.load(f)
# with open("../computed/bandit_start_16000_ucb_05_normsample.pkl", "rb") as f:
#     data_beam_05 = pickle.load(f)
# data_beam["scores"]["bandit (0, 0.5)"] = data_beam_05["scores"]["bandit (0, 0.5)"]
# data_sample["scores"]["bandit (0, 0.5)"] = data_sample_05["scores"]["bandit (0, 0.5)"]

# print(data_sample["scores"].keys())
METHOD_TO_NAME = {
    # 'bandit (0, 0.5)': "Bandit (exploit)",
    'bandit (0, 1.0)': "Bandit (default)",
    'random baseline': "Random",
    'top-k sum logprob': "LogProb Sum",
    'top-k avg logprob': "LogProb Avg",
}

for name, data in [("beam", data_beam), ("sample", data_sample)]:
    for mode in ["avg_score", "win_rate"]:
        plt.figure(figsize=(4, 2))
        for method in METHOD_TO_NAME:
            data_local = zip(*data["scores"][method][mode])
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
            # plt.ylim(0.1, None)
        elif mode == "avg_score":
            plt.gca().set_yticklabels(["{:,.1f}".format(x) for x in plt.gca().get_yticks()])
            # plt.ylim(78.5, 80.5)

        plt.legend(
            # vertical spacing
            labelspacing=0.2,
        )
        utils_figs.turn_off_spines()
        plt.tight_layout(pad=0.1)
        plt.savefig(f"../figures/52-bandit_{mode}_{name}.pdf")