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

from lib import utils


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
          h5py.File(Path(work_dir) / split / (utils.CANDIDATES_FILENAME +f"_{args.generation_mode}" +  ".h5")) as candidates_h5):
        candidates_h5ds = candidates_h5[utils.CANDIDATES_TEXT_H5DS_NAME]
        for i, data_line in enumerate(data_file):
            data = json.loads(data_line)
            if ((lang_pair == "all" or data["langs"] == lang_pair) and
                candidates_h5ds[i][0]):
                data_indices.append(i)

    return data_indices


def load_scores_and_similarities(
    data_path, work_dir, split, model_class_name):
    data_idxs = get_data_indices(
        args.data_dir, args.work_dir, split, args.lang_pair)

    split_work_dir = Path(args.work_dir) / split


    model_names = ([f"{args.model_class_name}"])

    num_metrics = len(model_names)
    with (h5py.File((split_work_dir / (utils.CANDIDATES_FILENAME+ f"_{args.generation_mode}")).with_suffix(".h5")) as cand_h5,
          h5py.File((split_work_dir / (utils.LOGPROBS_FILENAME_BASE + f"_{args.generation_mode}")).with_suffix(".h5")) as logprobs_h5,
          h5py.File((split_work_dir / (utils.SIMILARITIES_FILENAME_BASE + "cosine" f"_{args.generation_mode}")).with_suffix(".h5")) as sim_h5):
        counts_h5ds = cand_h5[utils.CANDIDATES_COUNTS_H5DS_NAME]
        sim_h5ds = sim_h5[utils.SIMILARITIES_H5DS_NAME]

        max_cands = counts_h5ds.shape[1]
        scores = np.zeros((len(data_idxs), max_cands, num_metrics))

        # Fetch all scores
        model_metric_idxs = np.arange(len(model_names))
        for metric_idx, model_name in zip(model_metric_idxs, model_names):

            h5_filename = split_work_dir / f"scores_comet_{model_name}_{args.generation_mode}.h5"
            with h5py.File(h5_filename) as scores_h5:
                scores_h5ds = scores_h5["scoresmodels-oxygen_24"] #[utils.COMET_SCORES_H5DS_NAME]

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

        breakpoint()
        return instance_scores, sims, counts, sum_logprobs, avg_logprobs


def main(args):
    np.random.seed(args.seed)
    work_dir = Path(args.work_dir)

    output_dir = work_dir / args.split / "gp_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = output_dir / f"{args.lang_pair}_{args.bandwidth}_{args.batch_size}_{args.generation_mode}.h5"

    all_scores, all_sims, all_counts, all_sum_logprobs, all_avg_logprobs = load_scores_and_similarities(
        args.data_dir, args.work_dir, args.split, args.model_class_name)

    INITIAL_SIZE = 10
    MAX_EVALS = 200

    log_data = {}
    for method in ["random", "random_deduped", "sum_logprob_first", "avg_logprob_first", "hill_climbing", "bayesopt"]:
        log_data[f"{method}_score"] = defaultdict(float)
        log_data[f"{method}_best_retrieved"] = defaultdict(float)
        log_data[f"{method}_scores"] = defaultdict(lambda: [0.0] * len(all_scores))

    for instance_idx, (scores, sims, counts, sum_logprobs, avg_logprobs) in enumerate(tqdm(zip(all_scores, all_sims, all_counts, all_sum_logprobs, all_avg_logprobs), total=len(all_sims))):
        scores = scores[:, -1]
        # For sampling without replacement for the baseline
        candidate_idxs = []
        for i in range(counts.size):
            candidate_idxs.extend([i] * int(counts[i]))
        np.random.shuffle(candidate_idxs)

        random_deduped_idxs = list(dict.fromkeys(candidate_idxs))

        sum_logprob_sorted_idxs = (-sum_logprobs[:scores.size]).argsort()
        avg_logprob_sorted_idxs = (-avg_logprobs[:scores.size]).argsort()

        best_idx = scores.argmax()

        # highest_logprob_idxs = list(list(zip(*sorted(zip(-np.array(logprobs), range(len(scores))))))[1][:MAX_EVALS])
        # baseline_total += scores[highest_logprob_idxs].max()

        all_idxs = np.arange(scores.shape[0])
        # known_idxs = list(np.random.choice(scores.shape[0], min(INITIAL_SIZE, all_idxs.shape[0]), replace=False))
        known_idxs = list(random_deduped_idxs[:INITIAL_SIZE])
        unknown_idxs = [x for x in all_idxs if x not in known_idxs]

        # For hill climbing baseline
        hc_known_idxs = list(known_idxs)
        hc_unknown_idxs = list(unknown_idxs)

        rbf_cov = np.exp(-(1 - sims.reshape(-1)) / (2 * args.bandwidth ** 2)).reshape(all_idxs.size, all_idxs.size)

        while len(known_idxs) < min(MAX_EVALS, all_idxs.shape[0]):
            log_data["random_score"][len(known_idxs)] += scores[candidate_idxs[:len(known_idxs)]].max()
            log_data["random_deduped_score"][len(known_idxs)] += scores[random_deduped_idxs[:len(known_idxs)]].max()
            log_data["hill_climbing_score"][len(known_idxs)] += scores[hc_known_idxs].max()
            log_data["sum_logprob_first_score"][len(known_idxs)] += scores[sum_logprob_sorted_idxs[:len(known_idxs)]].max()
            log_data["avg_logprob_first_score"][len(known_idxs)] += scores[avg_logprob_sorted_idxs[:len(known_idxs)]].max()
            log_data["bayesopt_score"][len(known_idxs)] += scores[known_idxs].max()

            log_data["random_best_retrieved"][len(known_idxs)] += int(candidate_idxs[scores[candidate_idxs[:len(known_idxs)]].argmax()] == best_idx)
            log_data["random_deduped_best_retrieved"][len(known_idxs)] += int(random_deduped_idxs[scores[random_deduped_idxs[:len(known_idxs)]].argmax()] == best_idx)
            log_data["hill_climbing_best_retrieved"][len(known_idxs)] += int(hc_known_idxs[scores[hc_known_idxs[:len(hc_known_idxs)]].argmax()] == best_idx)
            log_data["sum_logprob_first_best_retrieved"][len(known_idxs)] += int(sum_logprob_sorted_idxs[scores[sum_logprob_sorted_idxs[:len(known_idxs)]].argmax()] == best_idx)
            log_data["avg_logprob_first_best_retrieved"][len(known_idxs)] += int(avg_logprob_sorted_idxs[scores[avg_logprob_sorted_idxs[:len(known_idxs)]].argmax()] == best_idx)
            log_data["bayesopt_best_retrieved"][len(known_idxs)] += int(known_idxs[scores[known_idxs].argmax()] == best_idx)

            log_data["random_scores"][len(known_idxs)][instance_idx] = scores[candidate_idxs[:len(known_idxs)]].max()
            log_data["random_deduped_scores"][len(known_idxs)][instance_idx] = scores[random_deduped_idxs[:len(known_idxs)]].max()
            log_data["hill_climbing_scores"][len(known_idxs)][instance_idx] = scores[hc_known_idxs].max()
            log_data["sum_logprob_first_scores"][len(known_idxs)][instance_idx] = scores[sum_logprob_sorted_idxs[:len(known_idxs)]].max()
            log_data["avg_logprob_first_scores"][len(known_idxs)][instance_idx] = scores[avg_logprob_sorted_idxs[:len(known_idxs)]].max()
            log_data["bayesopt_scores"][len(known_idxs)][instance_idx] = scores[known_idxs].max()

            known_scores = scores[known_idxs]
            known_scores -= known_scores.mean()
            known_scores /= np.std(known_scores)
            # known_scores -= fixed_mean
            # known_scores /= fixed_std
            unknown_unknown_cov = rbf_cov[unknown_idxs][:, unknown_idxs]
            known_unknown_cov = rbf_cov[known_idxs][:, unknown_idxs]
            known_known_cov = rbf_cov[known_idxs][:, known_idxs]
            prior_cov = 0

            inverse_known_known_plus_prior = np.linalg.inv(known_known_cov)
            term_1 = np.matmul(inverse_known_known_plus_prior, known_unknown_cov)
            term_2 = np.matmul(known_unknown_cov.T, term_1)
            posterior_cov = unknown_unknown_cov - term_2
            mean_term_1 = np.matmul(inverse_known_known_plus_prior, known_scores)
            posterior_mean = np.matmul(known_unknown_cov.T, mean_term_1)
            posterior_var = posterior_cov.diagonal()
            best_score = known_scores.max()
            cdf = (norm.cdf(best_score, loc=posterior_mean, scale=posterior_var ** 0.5))
            # best_unknown_idx_idx = (1 - cdf).argmax()
            # Probability of improvement acquisition function
            try:
                best_idxs = np.array(unknown_idxs)[np.argpartition(cdf, min(args.batch_size, len(unknown_idxs)-1))[:args.batch_size]]
            except:
                import pdb; pdb.set_trace()

            # Expected improvement acquisition function
            # ei = ((posterior_mean * norm.cdf(best_score, loc=posterior_mean, scale=posterior_var ** 0.5)) +
            #       posterior_var ** 0.5 * norm.pdf(best_score, loc=posterior_mean, scale=posterior_var ** 0.5))
            z = (best_score - posterior_mean) / (posterior_var ** 0.5)
            ei = (
                posterior_var ** 0.5 *
                (z * norm.cdf(z) + norm.pdf(z))
            )
            best_idxs = np.array(unknown_idxs)[np.argpartition(ei, min(args.batch_size, len(unknown_idxs)-1))[:args.batch_size]]

            known_idxs = known_idxs + list(best_idxs)
            unknown_idxs = [x for x in all_idxs if x not in known_idxs]

            hc_best_idx = hc_known_idxs[scores[hc_known_idxs].argmax()]
            best_hc_idxs = np.array(hc_unknown_idxs)[np.argpartition(-sims[hc_best_idx][hc_unknown_idxs], min(args.batch_size, len(unknown_idxs)-1))[:args.batch_size]]
            hc_known_idxs = hc_known_idxs + list(best_hc_idxs)
            hc_unknown_idxs = [x for x in all_idxs if x not in hc_known_idxs]

        # while len(known_idxs) < min(MAX_EVALS, all_idxs.shape[0]):
        #     baseline_random_total[len(known_idxs)] += scores[candidate_idxs[:len(known_idxs)]].max()
        #     baseline_highest_logprob_total[len(known_idxs)] += scores[logprob_sorted_idxs[:len(known_idxs)]].max()
        #     bandit_total[len(known_idxs)] += scores[known_idxs].max()
        #     known_idxs.append(0)

        for total_cands in range(len(known_idxs), MAX_EVALS + 1):
            if total_cands % args.batch_size == 0:
                log_data["random_score"][total_cands] += scores[candidate_idxs[:total_cands]].max()
                log_data["random_deduped_score"][total_cands] += scores[random_deduped_idxs[:total_cands]].max()
                log_data["hill_climbing_score"][total_cands] += scores[hc_known_idxs].max()
                log_data["sum_logprob_first_score"][total_cands] += scores[sum_logprob_sorted_idxs[:total_cands]].max()
                log_data["avg_logprob_first_score"][total_cands] += scores[avg_logprob_sorted_idxs[:total_cands]].max()
                log_data["bayesopt_score"][total_cands] += scores[known_idxs].max()

                log_data["random_best_retrieved"][total_cands] += int(candidate_idxs[scores[candidate_idxs[:total_cands]].argmax()] == best_idx)
                log_data["random_deduped_best_retrieved"][total_cands] += int(random_deduped_idxs[scores[random_deduped_idxs[:total_cands]].argmax()] == best_idx)
                log_data["hill_climbing_best_retrieved"][total_cands] += int(hc_known_idxs[scores[hc_known_idxs].argmax()] == best_idx)
                log_data["sum_logprob_first_best_retrieved"][total_cands] += int(sum_logprob_sorted_idxs[scores[sum_logprob_sorted_idxs[:total_cands]].argmax()] == best_idx)
                log_data["avg_logprob_first_best_retrieved"][total_cands] += int(avg_logprob_sorted_idxs[scores[avg_logprob_sorted_idxs[:total_cands]].argmax()] == best_idx)
                log_data["bayesopt_best_retrieved"][total_cands] += int(known_idxs[scores[known_idxs].argmax()] == best_idx)

                log_data["random_scores"][total_cands][instance_idx] = scores[candidate_idxs[:len(known_idxs)]].max()
                log_data["random_deduped_scores"][total_cands][instance_idx] = scores[random_deduped_idxs[:len(known_idxs)]].max()
                log_data["hill_climbing_scores"][total_cands][instance_idx] = scores[hc_known_idxs].max()
                log_data["sum_logprob_first_scores"][total_cands][instance_idx] = scores[sum_logprob_sorted_idxs[:len(known_idxs)]].max()
                log_data["avg_logprob_first_scores"][total_cands][instance_idx] = scores[avg_logprob_sorted_idxs[:len(known_idxs)]].max()
                log_data["bayesopt_scores"][total_cands][instance_idx] = scores[known_idxs].max()

    for n in range(MAX_EVALS + 1):
        for method in ["random", "random_deduped", "sum_logprob_first", "avg_logprob_first", "hill_climbing", "bayesopt"]:
            for stat in ["score", "best_retrieved"]:
                log_data[f"{method}_{stat}"][n] /= len(all_scores)

    output_h5 = h5py.File(output_filename, 'w')

    for method in ["random", "random_deduped", "sum_logprob_first", "avg_logprob_first", "hill_climbing", "bayesopt"]:
        for stat in ["score", "best_retrieved"]:
            h5ds = output_h5.create_dataset(f"{method}_{stat}", (MAX_EVALS + 1,), float)
            h5ds[:] = np.array([log_data[f"{method}_{stat}"][n] for n in range(MAX_EVALS+1)])
        h5ds = output_h5.create_dataset(f"{method}_scores", (MAX_EVALS + 1, len(all_scores)), float)
        for n in range(MAX_EVALS+1):
            h5ds[n] = np.array(log_data[f"{method}_scores"][n])

        # for n in range(MAX_EVALS):


    # print(log_data["baseline_max_total"] / len(all_scores))
    # for k in log_data["bandit_total"]:
    #     print(k,
    #             log_data["bandit_total"][k] / log_data["num_instances"],
    #             log_data["baseline_random_total"][k] / log_data["num_instances"],
    #             log_data["baseline_random_deduped_total"][k] / log_data["num_instances"],
    #             log_data["baseline_hill_climbing_total"][k] / log_data["num_instances"],
    #             log_data["baseline_highest_logprob_total"][k] / log_data["num_instances"])


    # for k in log_data["bandit_total"]:
    #     print(k,
    #             log_data["bandit_best_retrieved"][k] / log_data["num_instances"],
    #             log_data["baseline_random_best_retrieved"][k] / log_data["num_instances"],
    #             log_data["baseline_random_deduped_best_retrieved"][k] / log_data["num_instances"],
    #             log_data["baseline_hill_climbing_best_retrieved"][k] / log_data["num_instances"],
    #             log_data["baseline_highest_logprob_best_retrieved"][k] / log_data["num_instances"])


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
        "--alpha", type=float, default=0.2, help="Confidence threshold.")

    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--generation_mode", type=str, default="sample", help="Either 'beam' or 'sample'.")

    args = parser.parse_args()
    main(args)

    # python scripts/run_simple_gp_bandit.py vilem/scripts/data output_emb models-oxygen all test_sample 0.25 10
