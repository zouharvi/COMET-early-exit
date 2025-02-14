import argparse
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm
from lib import utils

import torch
import torch.nn.functional as F

def main(args):
    for split in ["test_sample"]:
        split_work_dir = Path(args.work_dir) / split
        with (h5py.File((split_work_dir / f"{utils.CANDIDATES_FILENAME}_{args.generation_mode}").with_suffix(".h5")) as cand_h5,
              h5py.File((split_work_dir / (utils.LOGPROBS_FILENAME_BASE+ f"_{args.generation_mode}")).with_suffix(".h5"), "w") as sim_h5):
            logprobs_h5ds = cand_h5[utils.CANDIDATES_TOKEN_LOGPROBS_H5DS_NAME]
            sum_h5ds = sim_h5.create_dataset(
                utils.SUM_LOGPROBS_H5DS_NAME,
                logprobs_h5ds.shape,
                float)
            avg_h5ds = sim_h5.create_dataset(
                utils.AVG_LOGPROBS_H5DS_NAME,
                logprobs_h5ds.shape,
                float)

            for idx in tqdm(range(logprobs_h5ds.shape[0])):
                for cand_idx in range(logprobs_h5ds.shape[1]):
                    logprobs = logprobs_h5ds[idx, cand_idx]
                    sum_logprob = np.sum(logprobs)
                    avg_logprob = sum_logprob / len(logprobs)
                    sum_h5ds[idx, cand_idx] = sum_logprob
                    avg_h5ds[idx, cand_idx] = avg_logprob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "data_dir", help="Data directory generated by the pipeline from vilem/scripts.")

    parser.add_argument(
        "work_dir", help="Working directory for all steps. "
                         "Will be created if doesn't exist.")
    
    parser.add_argument(
        "--generation_mode", type=str, default="sample", help="Either 'beam' or 'sample'.")
    args = parser.parse_args()
    main(args)