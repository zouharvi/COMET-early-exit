# %%

import matplotlib.pyplot as plt
import utils_figs
import pickle

# TODO: plot ablation in the appendix

with open("../computed/bandit_ucb_ablation_16000_norm_start0sample.pkl", "rb") as f:
    data_sample = pickle.load(f)
with open("../computed/bandit_ucb_ablation_14324_norm_start0beam.pkl", "rb") as f:
    data_beam = pickle.load(f)

data_sample["scores"].keys()

# %%

METHOD_TO_CONFIG = {
    # 'bandit (0, 0.0)': dict(name=r"Bandit ($\gamma=0.0$)", color=utils_figs.COLORS[0], alpha=1),
    'bandit (0, 0.5)': dict(name=r"Bandit ($\gamma=0.5$)", color=utils_figs.COLORS[0], alpha=1.0),
    'bandit (0, 1.0)': dict(name=r"Bandit ($\gamma=1.0$)", color=utils_figs.COLORS[0], alpha=0.6),
    'bandit (0, 2.0)': dict(name=r"Bandit ($\gamma=2.0$)", color=utils_figs.COLORS[0], alpha=0.2),
    'random baseline': dict(name="Random", color=utils_figs.COLORS[1], alpha=1),
    # 'top-k sum logprob': dict(name="LogProb Sum", color=utils_figs.COLORS[2], alpha=1),
    # 'top-k avg logprob': dict(name="LogProb Avg", color=utils_figs.COLORS[3], alpha=1),
}

for name, data in [("beam", data_beam), ("sample", data_sample)]:
    for mode in ["avg_score", "win_rate"]:
        plt.figure(figsize=(4, 3))
        for method in METHOD_TO_CONFIG:
            data_local = zip(*data["scores"][method][mode])
            data_local = [x for x in data_local if x[0] > 0.1]
            plt.plot(
                [x[0] for x in data_local],
                [x[1] for x in data_local],
                label=METHOD_TO_CONFIG[method]["name"],
                color=METHOD_TO_CONFIG[method]["color"],
                alpha=METHOD_TO_CONFIG[method]["alpha"],
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


        plt.legend(
            # vertical spacing
            labelspacing=0.2,
        )
        utils_figs.turn_off_spines()
        plt.tight_layout(pad=0.1)
        plt.savefig(f"../figures/54-bandit_ablation_gamma_{mode}_{name}.pdf")