# %%

import json
import numpy as np
import scipy.stats

# data = json.load(open("../computed/eval_helium_mcd_02.json", "r"))
# data = json.load(open("../computed/eval_helium_mcd_05.json", "r"))
# data = json.load(open("../computed/eval_helium_mcd_10.json", "r"))
data = json.load(open("../computed/eval_helium_mcd_50.json", "r"))

# %%

arr_score = np.array([x["mcd_score"] for x in data])
arr_human = np.array([x["human"] for x in data])

# correlation between arr_conf and |arr_score-arr_human|
for conf_key in ["mcd_std", "mcd_mad"]:
    hum_corr = scipy.stats.pearsonr(
        arr_score, arr_human,
    ).correlation
    error_corr = scipy.stats.pearsonr(
        np.array([x[conf_key] for x in data]),
        np.abs(arr_score-arr_human),
    ).correlation
    print(f"{conf_key}: {hum_corr:.3f} {error_corr:.3f}")

# %% 
"""
scp euler:/cluster/work/sachan/vilem/COMET-early-exit/logs/eval_helium_mcd_02.out computed/eval_helium_mcd_02.json
scp euler:/cluster/work/sachan/vilem/COMET-early-exit/logs/eval_helium_mcd_05.out computed/eval_helium_mcd_05.json
scp euler:/cluster/work/sachan/vilem/COMET-early-exit/logs/eval_helium_mcd_10.out computed/eval_helium_mcd_10.json
scp euler:/cluster/work/sachan/vilem/COMET-early-exit/logs/eval_helium_mcd_50.out computed/eval_helium_mcd_50.json
scp euler:/cluster/work/sachan/vilem/COMET-early-exit/logs/eval_helium_mcd_100.out computed/eval_helium_mcd_100.json
"""
