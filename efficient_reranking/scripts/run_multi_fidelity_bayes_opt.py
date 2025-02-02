import argparse
from collections import defaultdict
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

DEV_CORR = {
    "avg_logprob": 0.30261028985941163,
    "S": 0.6722579350773578,
    "M": 0.7134984155295572
}


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
                candidates_h5ds[i][0] and
                data["langs"] != "cs-uk"):
                data_indices.append(i)

    return data_indices


def load_scores_and_similarities(
    data_path, work_dir, split, model_class_name):
    data_idxs = get_data_indices(
        args.data_dir, args.work_dir, split, args.lang_pair)

    split_work_dir = Path(args.work_dir) / split

    model_names = ([f"{args.model_class_name}-{size}" for size in ("S", "M", "L")] +
                   ["wmt22-cometkiwi-da"])

    num_metrics = len(model_names)
    with (h5py.File((split_work_dir / utils.CANDIDATES_FILENAME).with_suffix(".h5")) as cand_h5,
          h5py.File((split_work_dir / utils.LOGPROBS_FILENAME_BASE).with_suffix(".h5")) as logprobs_h5,
          h5py.File((split_work_dir / (utils.SIMILARITIES_FILENAME_BASE + "cosine")).with_suffix(".h5")) as sim_h5):
        counts_h5ds = cand_h5[utils.CANDIDATES_COUNTS_H5DS_NAME]
        sim_h5ds = sim_h5[utils.SIMILARITIES_H5DS_NAME]

        max_cands = counts_h5ds.shape[1]
        scores = np.zeros((len(data_idxs), max_cands, num_metrics))

        # Fetch all scores
        model_metric_idxs = np.arange(len(model_names))
        for metric_idx, model_name in zip(model_metric_idxs, model_names):
            h5_filename = split_work_dir / f"scores_comet_{model_name}.h5"
            with h5py.File(h5_filename) as scores_h5:
                scores_h5ds = scores_h5[utils.COMET_SCORES_H5DS_NAME]
                scores[:, :, metric_idx] = scores_h5ds[data_idxs]

        sum_logprobs = logprobs_h5[utils.SUM_LOGPROBS_H5DS_NAME][data_idxs]
        avg_logprobs = logprobs_h5[utils.AVG_LOGPROBS_H5DS_NAME][data_idxs]
        counts = counts_h5ds[data_idxs]

        # Break out the big score matrix into a list of scores per instance
        # due to varying numbers of candidates per instance
        instance_scores = []
        sims = []
        for idx, data_idx in enumerate(tqdm(data_idxs)):
            num_cands = (counts_h5ds[data_idx] > 0).sum()
            instance_scores.append(scores[idx, :num_cands])
            sims.append(sim_h5ds[data_idx].reshape(max_cands, max_cands)[:num_cands, :num_cands])

        return instance_scores, sims, counts, sum_logprobs, avg_logprobs


def normalize(arr):
    return (arr - arr.mean()) / arr.std()


def main(args):
    np.random.seed(args.seed)
    work_dir = Path(args.work_dir)

    all_scores, all_sims, all_counts, all_sum_logprobs, all_avg_logprobs = load_scores_and_similarities(
        args.data_dir, args.work_dir, args.split, args.model_class_name)

    INITIAL_SIZE = 10
    MAX_EVALS = 200

    log_data = {}
    for method in ["bayesopt", "proxy_first"]:
        log_data[f"{method}_score"] = defaultdict(float)
        log_data[f"{method}_best_retrieved"] = defaultdict(float)
        log_data[f"{method}_scores"] = defaultdict(lambda: [0.0] * len(all_scores))

    # corrs = []
    # if args.use_dev_correlation:
    #     m_1_scores = []
    #     m_star_scores = []
    #     for scores, counts, avg_logprobs in zip(all_scores, all_counts, all_avg_logprobs):
    #         if args.metric == "S":
    #             m_scores_orig = scores[:, 1]
    #         elif args.metric == "M":
    #             m_scores_orig = scores[:, 2]
    #         elif args.metric == "avg_logprob":
    #             m_scores_orig = avg_logprobs[:(counts > 0).sum()]
    #         else:
    #             raise ValueError(f"Unknown metric '{args.metric}'")
    #         m_1_scores.append(normalize(m_scores_orig))
    #         m_star_scores.append(normalize(scores[:, -1]))
    #     dev_corr = pearsonr(np.concatenate(m_1_scores), np.concatenate(m_star_scores))
    #     print(dev_corr)
    #     sys.exit()

    # all_scores = all_scores[:100]

    for instance_idx, (scores, sims, counts, sum_logprobs, avg_logprobs) in enumerate(tqdm(zip(all_scores, all_sims, all_counts, all_sum_logprobs, all_avg_logprobs))):
        if args.metric == "S":
            m_scores_orig = scores[:, 1]
        elif args.metric == "M":
            m_scores_orig = scores[:, 2]
        elif args.metric == "avg_logprob":
            m_scores_orig = avg_logprobs[:(counts > 0).sum()]
        else:
            raise ValueError(f"Unknown metric '{args.metric}'")

        candidate_idxs = []
        for i in range(counts.size):
            candidate_idxs.extend([i] * int(counts[i]))
        np.random.shuffle(candidate_idxs)
        random_deduped_idxs = list(dict.fromkeys(candidate_idxs))

        m_1_subset_idxs = random_deduped_idxs[:args.num_proxy_evals]

        # m_scores -= m_scores.mean()
        # m_scores /= m_scores.std()
        m_star_scores = scores[:, -1]

        all_idxs = np.arange(m_star_scores.shape[0])

        m_1_sorted_idxs = list(list(zip(*sorted(zip(-m_scores_orig[m_1_subset_idxs], m_1_subset_idxs))))[1])
        m_1_first_sorted_idxs = m_1_sorted_idxs + [idx for idx in random_deduped_idxs if idx not in set(m_1_sorted_idxs)]

        known_idxs = m_1_sorted_idxs[:INITIAL_SIZE]
        unknown_idxs = [x for x in all_idxs if x not in known_idxs]
        # Elements for which m_1 scores are known and m_star_scores are unknown.
        # Need a better name for this
        # m_1_used_idxs = [x for x in m_1_sorted_idxs if x not in known_idxs]
        m_1_used_idxs = list(m_1_sorted_idxs)

        rbf_cov = np.exp(-(1 - sims.reshape(-1)) / (2 * args.bandwidth ** 2)).reshape(all_idxs.size, all_idxs.size)

        while len(known_idxs) < min(MAX_EVALS, all_idxs.shape[0]):
            log_data["proxy_first_score"][len(known_idxs)] += m_star_scores[m_1_first_sorted_idxs[:len(known_idxs)]].max()
            log_data["bayesopt_score"][len(known_idxs)] += m_star_scores[known_idxs].max()

            log_data["proxy_first_scores"][len(known_idxs)][instance_idx] = m_star_scores[m_1_first_sorted_idxs[:len(known_idxs)]].max()
            log_data["bayesopt_scores"][len(known_idxs)][instance_idx] = m_star_scores[known_idxs].max()

            known_scores_m_1 = normalize(m_scores_orig[m_1_used_idxs])
            known_scores_m_star = normalize(m_star_scores[known_idxs])

            if args.use_dev_correlation:
                metrics_corr = DEV_CORR[args.metric]
            else:
                metrics_corr = pearsonr(normalize(m_scores_orig[known_idxs]), known_scores_m_star).statistic

            known_scores = np.concatenate([known_scores_m_1, known_scores_m_star])

            unknown_unknown_cov = rbf_cov[unknown_idxs][:, unknown_idxs]

            # known_scores = np.concatenate([known_scores_m_1[unknown_idxs], known_scores_m_star])

            # known_unknown_cov_m_1 = rbf_cov[unknown_idxs][:, unknown_idxs] * metrics_corr
            # known_unknown_cov_m_star = rbf_cov[known_idxs][:, unknown_idxs]
            # known_unknown_cov = np.concatenate([known_unknown_cov_m_1, known_unknown_cov_m_star])

            # known_known_cov_m_star = rbf_cov[known_idxs][:, known_idxs]
            # known_known_cov_m_1_m_star = rbf_cov[known_idxs][:, unknown_idxs] * metrics_corr
            # known_known_cov_m_1 = rbf_cov[unknown_idxs][:, unknown_idxs]
            # known_known_cov = np.concatenate([np.concatenate([known_known_cov_m_1, known_known_cov_m_1_m_star]), np.concatenate([known_known_cov_m_1_m_star.T, known_known_cov_m_star])], axis=1)

            known_scores = np.concatenate([m_scores_orig[m_1_used_idxs], known_scores_m_star])

            known_unknown_cov_m_1 = rbf_cov[m_1_used_idxs][:, unknown_idxs] * metrics_corr
            known_unknown_cov_m_star = rbf_cov[known_idxs][:, unknown_idxs]
            known_unknown_cov = np.concatenate([known_unknown_cov_m_1, known_unknown_cov_m_star])

            known_known_cov_m_star = rbf_cov[known_idxs][:, known_idxs]
            known_known_cov_m_1_m_star = rbf_cov[known_idxs][:, m_1_used_idxs] * metrics_corr
            known_known_cov_m_1 = rbf_cov[m_1_used_idxs][:, m_1_used_idxs]
            known_known_cov = np.concatenate([np.concatenate([known_known_cov_m_1, known_known_cov_m_1_m_star]), np.concatenate([known_known_cov_m_1_m_star.T, known_known_cov_m_star])], axis=1)

            inverse_known_known_plus_prior = np.linalg.inv(known_known_cov)
            term_1 = np.matmul(inverse_known_known_plus_prior, known_unknown_cov)
            term_2 = np.matmul(known_unknown_cov.T, term_1)
            posterior_cov = unknown_unknown_cov - term_2
            mean_term_1 = np.matmul(inverse_known_known_plus_prior, known_scores)
            posterior_mean = np.matmul(known_unknown_cov.T, mean_term_1)
            posterior_var = posterior_cov.diagonal()

            best_score = known_scores.max()
            z = (best_score - posterior_mean) / (posterior_var ** 0.5)
            ei = (
                posterior_var ** 0.5 *
                (z * norm.cdf(z) + norm.pdf(z))
            )
            best_idxs = np.array(unknown_idxs)[np.argpartition(ei, min(args.batch_size, len(unknown_idxs)-1))[:args.batch_size]]

            known_idxs = known_idxs + list(best_idxs)
            unknown_idxs = [x for x in all_idxs if x not in known_idxs]
            # m_1_used_idxs = [x for x in m_1_sorted_idxs if x not in known_idxs]


        for total_cands in range(len(known_idxs), MAX_EVALS + 1):
            log_data["proxy_first_score"][total_cands] += m_star_scores[m_1_first_sorted_idxs[:total_cands]].max()
            log_data["bayesopt_score"][total_cands] += m_star_scores[known_idxs].max()

            log_data["proxy_first_scores"][total_cands][instance_idx] += m_star_scores[m_1_first_sorted_idxs[:total_cands]].max()
            log_data["bayesopt_scores"][total_cands][instance_idx] += m_star_scores[known_idxs].max()

            # print(total_cands, m_star_scores[known_idxs].max())

    output_dir = work_dir / args.split / "proxy_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.use_dev_correlation:
        output_filename = output_dir / f"{args.lang_pair}_{args.bandwidth}_{args.batch_size}_{args.metric}_{args.num_proxy_evals}_devcorr.h5"
    else:
        output_filename = output_dir / f"{args.lang_pair}_{args.bandwidth}_{args.batch_size}_{args.metric}_{args.num_proxy_evals}.h5"

    output_h5 = h5py.File(output_filename, 'w')

    for n in range(MAX_EVALS + 1):
        for method in ["proxy_first", "bayesopt"]:
            for stat in ["score"]:
                log_data[f"{method}_{stat}"][n] /= len(all_scores)

    for method in ["proxy_first", "bayesopt"]:
        for stat in ["score"]:
            h5ds = output_h5.create_dataset(f"{method}_{stat}", (MAX_EVALS + 1,), float)
            h5ds[:] = np.array([log_data[f"{method}_{stat}"][n] for n in range(MAX_EVALS+1)])
        h5ds = output_h5.create_dataset(f"{method}_scores", (MAX_EVALS + 1, len(all_scores)), float)
        for n in range(MAX_EVALS+1):
            h5ds[n] = np.array(log_data[f"{method}_scores"][n])

    for k in sorted(log_data["proxy_first_score"]):
        if k % 10 == 0:
            print(k, log_data["proxy_first_score"][k], log_data["bayesopt_score"][k])

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
        "split", help="Data split.")

    parser.add_argument(
        "bandwidth", type=float, help="RBF bandwidth parameter.")

    parser.add_argument(
        "batch_size", type=int, help="Bayesopt update batch size.")

    parser.add_argument(
        "metric", type=str, help="")

    parser.add_argument(
        "num_proxy_evals", type=int, help="")

    parser.add_argument(
        "--alpha", type=float, default=0.2, help="Confidence threshold.")

    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed.")

    parser.add_argument(
        "--use_dev_correlation", action="store_true", help="")


    args = parser.parse_args()
    main(args)
