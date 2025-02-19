# %%

import matplotlib.pyplot as plt
import utils_figs
import pickle

# TODO: plot ablation in the appendix

with open("../computed/bandit_start_ablation_16000_norm_ucb10sample.pkl", "rb") as f:
    data_sample = pickle.load(f)
with open("../computed/bandit_start_ablation_16000_norm_ucb10sample.pkl", "rb") as f:
    data_beam = pickle.load(f)

print(data_sample["scores"].keys())

METHOD_TO_CONFIG = {
    'bandit (0, 1.0)': dict(name="Bandit ($\\geq$layer 1)", color=utils_figs.COLORS[0], alpha=1),
    'bandit (4, 1.0)': dict(name="Bandit ($\\geq$layer 5)", color=utils_figs.COLORS[0], alpha=0.5),
    'bandit (12, 1.0)': dict(name="Bandit ($\\geq$layer 13)", color=utils_figs.COLORS[0], alpha=0.2),
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
            plt.ylim(0.1, None)
            plt.gca().set_yticklabels(["{:,.0%}".format(x) for x in plt.gca().get_yticks()])
        elif mode == "avg_score":
            plt.ylim(78.5, 80.5)
            plt.gca().set_yticklabels(["{:,.1f}".format(x) for x in plt.gca().get_yticks()])


        plt.legend(
            # vertical spacing
            labelspacing=0.2,
        )
        utils_figs.turn_off_spines()
        plt.tight_layout(pad=0.1)
        plt.savefig(f"../figures/53-bandit_ablation_start_{mode}_{name}.pdf")