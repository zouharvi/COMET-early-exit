import argparse
import json
import logging
import sys

import pandas as pd

from pathlib import Path
from tqdm import tqdm

# Logging format borrowed from Fairseq.
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

import numpy as np
import h5py
from scipy.stats import pearsonr, norm, kendalltau
from pprint import pprint

from efficient_reranking.lib import utils


def conditional_mean_and_covar(known_values, mean, covar):
    num_known_columns = known_values.shape[1]
    # y refers to the conditioned variables, x the condition variables
    x_mean = mean[:num_known_columns]
    y_mean = mean[num_known_columns:]
    xx_covar = covar[:num_known_columns,:num_known_columns]
    yx_covar = covar[num_known_columns:,:num_known_columns]
    xy_covar = yx_covar.T
    yy_covar = covar[num_known_columns:,num_known_columns:]
    y_given_x_mean = np.expand_dims(y_mean, 1) + np.dot(np.dot(yx_covar, np.linalg.inv(xx_covar)), (known_values - x_mean).T)
    # return y_given_x_mean.T
    y_given_x_covar = yy_covar - np.dot(yx_covar, np.dot(np.linalg.inv(xx_covar), xy_covar))
    return y_given_x_mean.T, y_given_x_covar


def get_data_indices(data_dir, work_dir, split, lang_pair):
    # Only include instances which match the desired language pair and have candidates
    # (some candidates failed due to OOM).
    data_indices = []
    data_path = Path(data_dir) / "jsonl" / f"{split}.jsonl"
    with (open(data_path) as data_file,
          h5py.File(Path(work_dir) / split / (utils.CANDIDATES_FILENAME + ".h5")) as candidates_h5):
        candidates_h5ds = candidates_h5[utils.CANDIDATES_TEXT_H5DS_NAME]
        for i, data_line in enumerate(data_file):
            data = json.loads(data_line)
            if ((lang_pair == "all" or data["langs"] == lang_pair) and
                candidates_h5ds[i][0]):
                data_indices.append(i)

    return data_indices


def load_scores(data_path,
                work_dir,
                split,
                model_class_name,
                get_avg_logprob,
                get_sum_logprob):
    data_idxs = get_data_indices(
        args.data_dir, args.work_dir, split, args.lang_pair)

    split_work_dir = Path(args.work_dir) / split

    model_names = ([f"{args.model_class_name}-{size}" for size in ("S", "M", "L")] +
                   ["wmt22-cometkiwi-da"])

    num_metrics = args.avg_logprob + args.sum_logprob + len(model_names)
    with h5py.File((Path(args.work_dir) / split / utils.CANDIDATES_FILENAME).with_suffix(".h5")) as h5_file:
        if args.avg_logprob or args.sum_logprob:
            token_logprobs = h5_file[utils.CANDIDATES_TOKEN_LOGPROBS_H5DS_NAME][:]
        counts_h5ds = h5_file[utils.CANDIDATES_COUNTS_H5DS_NAME]
        scores = np.zeros((len(data_idxs), counts_h5ds.shape[1], num_metrics))

        # Fetch all scores
        model_metric_idxs = np.arange(len(model_names)) + args.avg_logprob + args.sum_logprob
        for metric_idx, model_name in zip(model_metric_idxs, model_names):
            h5_filename = split_work_dir / f"scores_comet_{model_name}.h5"
            with h5py.File(h5_filename) as h5_file:
                scores_h5ds = h5_file[utils.COMET_SCORES_H5DS_NAME]
                scores[:, :, metric_idx] = scores_h5ds[data_idxs]

        # Break out the big score matrix into a list of scores per instance
        # due to varying numbers of candidates per instance
        instance_scores = []
        for idx, data_idx in enumerate(data_idxs):
            num_cands = (counts_h5ds[data_idx] > 0).sum()
            instance_scores.append(scores[idx, :num_cands])

    return instance_scores, model_names


def main(args):
    np.random.seed(args.seed)
    work_dir = Path(args.work_dir)

    logging.info("Fetching scores")

    if args.avg_logprob or args.sum_logprob:
        raise ValueError("Not implemented")

    all_dev_scores, score_names = load_scores(
        args.data_dir, args.work_dir, "dev", args.model_class_name,
        args.avg_logprob, args.sum_logprob)
    all_test_scores_orig, _ = load_scores(
        args.data_dir, args.work_dir, "test_small", args.model_class_name,
        args.avg_logprob, args.sum_logprob)

    if args.zero_mean:
        dev_scores_train = [scores - scores.mean(0) for scores in all_dev_scores]
        all_test_scores = [scores - scores.mean(0) for scores in all_test_scores_orig]
    else:
        dev_scores_train = all_dev_scores
        all_test_scores = all_test_scores_orig

    logging.info("Fitting multivariate gaussian")
    dev_scores_train_stacked = np.concatenate(dev_scores_train, axis=0)

    mean = dev_scores_train_stacked.mean(0)
    cov = np.cov(dev_scores_train_stacked.T)

    logging.info("Running pruning prediction algorithm")
    total_score_random = 0
    total_score_baseline = 0
    total_score_pruning = 0
    best_candidate_chosen = 0
    runs_per_metric = [0] * all_test_scores[0].shape[1]

    for test_scores, test_scores_orig in zip(all_test_scores, all_test_scores_orig):
        total_score_baseline += test_scores_orig[:, -1].max()
        total_score_random += np.random.choice(test_scores_orig[:, -1])
        best_candidate_idx = test_scores_orig[:, -1].argmax()

        candidate_idxs = np.arange(test_scores.shape[0])
        for metric_idx in range(test_scores.shape[1] - 1):
            runs_per_metric[metric_idx] += candidate_idxs.size
            known_values = test_scores[:, :metric_idx+1]
            y_given_x_mean, y_given_x_cov = conditional_mean_and_covar(known_values, mean, cov)
            estimated_best_idx = y_given_x_mean[:, -1].argmax()
            estimated_best_mean = y_given_x_mean[estimated_best_idx, -1]
            cdf = norm.cdf(y_given_x_mean[:, -1] - estimated_best_mean, scale=(y_given_x_cov[-1, -1])**0.5)
            test_scores = test_scores[cdf >= args.alpha, :]
            candidate_idxs = candidate_idxs[cdf >= args.alpha]

        runs_per_metric[-1] += candidate_idxs.size
        predicted_candidate_idx = candidate_idxs[test_scores_orig[candidate_idxs, -1].argmax()]
        total_score_pruning += test_scores_orig[predicted_candidate_idx, -1].max()
        if predicted_candidate_idx == best_candidate_idx:
            best_candidate_chosen += 1

    statistic_names = [
        "Final score - random",
        "Final score - baseline",
        "Final score -  pruning",
    ]
    statistic_names += [f"Metric runs - {score_name}" for score_name in score_names]

    results_rows = []
    results_rows.append([f"Final score - random", total_score_random / len(all_test_scores)])
    results_rows.append([f"Final score - baseline", total_score_baseline / len(all_test_scores)])
    results_rows.append([f"Final score - pruning", total_score_pruning / len(all_test_scores)])
    for score_name, num_runs in zip(score_names, runs_per_metric):
        results_rows.append([f"Metric runs - {score_name}", num_runs / len(all_test_scores)])
    results_rows.append([f"Top-1 returned", best_candidate_chosen / len(all_test_scores)])
    results_df = pd.DataFrame(data=results_rows, columns=["Statistic", "Value"])
    print(results_df.to_string(index=False))
    # Print just the values for ease of pasting into a spreadsheet
    print(results_df["Value"].to_string(index=False))

    # print((np.array(runs_per_metric) * np.array([0.7, 1.3, 2.7, 5.7]) / len(all_test_scores)).sum())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "data_dir", help="Data directory generated by the pipeline from vilem/scripts.")

    parser.add_argument(
        "work_dir", help="Working directory for all steps. "
                         "Will be created if doesn't exist.")

    parser.add_argument(
        "model_class_name", help="Name of the model class, e.g. 'quern', 'skintle'.")

    parser.add_argument(
        "lang_pair", help="E.g. 'en-cs'. 'all' for all language pairs.")

    parser.add_argument(
        "--alpha", type=float, default=0.2, help="Confidence threshold.")

    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed.")

    parser.add_argument(
        "--zero_mean", action="store_true",
        help="Whether to zero-mean each metric ")

    parser.add_argument(
        "--avg_logprob", action="store_true",
        help="Whether to use average token log probability as a metric.")

    parser.add_argument(
        "--sum_logprob", action="store_true",
        help="Whether to use total token log probability as a metric.")

    args = parser.parse_args()
    main(args)
