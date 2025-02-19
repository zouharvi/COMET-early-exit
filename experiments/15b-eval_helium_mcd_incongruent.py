# %%

import json
import numpy as np
import scipy.stats
import csv


data_helium = list(csv.DictReader(open("../computed/test_sodium.csv", "r")))
arr_score = [float(x["predicted"]) for x in data_helium]

for fname in [
    "../computed/eval_helium_mcd_02.json",
    "../computed/eval_helium_mcd_05.json",
    "../computed/eval_helium_mcd_10.json",
    "../computed/eval_helium_mcd_50.json",
    "../computed/eval_helium_mcd_100.json",
]:
    print(fname)
    data = json.load(open(fname, "r"))

    assert len(data) == len(data_helium)
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
scp euler:/cluster/work/sachan/vilem/COMET-early-exit/data/csv/test_sodium.csv computed/test_sodium.csv
"""
