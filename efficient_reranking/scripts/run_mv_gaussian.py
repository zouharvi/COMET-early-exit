import argparse
import json
import logging
import sys

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

# f = h5py.File("/home/jncc3/efficient_reranking/output/deen/validation/data.h5")
# all_scores = []
# # for h5ds_name in ["scores_comet_quern-minilm",
# #                   "scores_comet_quern-bert-base-multilingual-cased",
# #                   "scores_comet_quern-xlm-roberta-base",
# #                   "scores_comet_quern-xlm-roberta-large"]:
# for h5ds_name in [f"scores_comet_wmt22-cometkiwi-da_{i}" for i in range(1, 25)]:
#     all_scores.append(f[h5ds_name][:])
# all_scores = np.stack(all_scores)
# unnormalized_scores = all_scores * 1
# all_scores -= all_scores.mean(axis=-1,keepdims=True)
# # all_scores /= all_scores.std(axis=-1,keepdims=True)

# num_train_instances = int(all_scores.shape[1] * 0.8)
# train_scores = all_scores[:, :num_train_instances, :]
# train_scores_reshaped = train_scores.reshape(train_scores.shape[0], -1)
# test_scores = all_scores[:, num_train_instances:, :]
# unnormalized_train_scores = unnormalized_scores[:, :num_train_instances, :]
# unnormalized_test_scores = unnormalized_scores[:, num_train_instances:, :]

# # all_scores = all_scores.reshape(all_scores.shape[0], -1)
# # print(np.corrcoef(all_scores.reshape(all_scores.shape[0], -1)))


# mean = train_scores_reshaped.mean(-1)
# cov = np.cov(train_scores_reshaped)
# baseline_total = 0
# pruning_total = 0
# random_total = 0
# num_metric_runs = [0] * (test_scores.shape[0])

# ALPHA = 0.2
# for instance_idx in range(test_scores.shape[1]):
#     baseline_total += unnormalized_test_scores[-1, instance_idx].max()
#     random_total += np.random.choice(unnormalized_test_scores[-1, instance_idx])
#     # print(f"INSTANCE {instance_idx}")
#     unnormalized_instance_scores = unnormalized_test_scores[:, instance_idx]
#     instance_scores = test_scores[:, instance_idx]

#     for metric_idx in range(instance_scores.shape[0]-1):
#         num_metric_runs[metric_idx] += instance_scores.shape[1]
#         # print(f"METRIC {metric_idx+1}")
#         known_values = instance_scores[:metric_idx+1].T
#         y_given_x_mean, y_given_x_cov = conditional_mean_and_covar(known_values, mean, cov)
#         estimated_best_idx = y_given_x_mean[:, -1].argmax()
#         estimated_best_mean = y_given_x_mean[estimated_best_idx][-1]
#         cdf = norm.cdf(y_given_x_mean[:, -1] - estimated_best_mean, scale=(y_given_x_cov[-1, -1] * 2)**0.5)
#         instance_scores = instance_scores[:, cdf >= ALPHA]
#         unnormalized_instance_scores = unnormalized_instance_scores[:, cdf >= ALPHA]
#     num_metric_runs[-1] += unnormalized_instance_scores.shape[1]
#     pruning_total += unnormalized_instance_scores[-1].max()

# print(baseline_total / test_scores.shape[1])
# print(pruning_total / test_scores.shape[1])
# print(random_total / test_scores.shape[1])
# print([x / test_scores.shape[1] for x in num_metric_runs])


def get_data_indices(data_dir, work_dir, split, model_class_name, lang_pair):
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


def main(args):
    np.random.seed(args.seed)
    work_dir = Path(args.work_dir)

    # (julius) Haven't run the test set yet so just splitting dev set into dev/test.
    logging.info("Fetching scores")

    split = "dev"
    data_indices = get_data_indices(
        args.data_dir, args.work_dir, split, args.model_class_name, args.lang_pair)

    score_names = []
    scores = []
    model_names = ([f"{args.model_class_name}-{size}" for size in ("S", "M", "L")] +
                   ["wmt22-cometkiwi-da"])
    for model_name in model_names:
        score_names.append(model_name)
        split_work_dir = Path(work_dir) / split
        h5_filename = split_work_dir / f"scores_comet_{model_name}.h5"
        with h5py.File(h5_filename) as h5_file:
            scores.append(h5_file[utils.COMET_SCORES_H5DS_NAME][data_indices])

    # Shape of (# of scores, # of instances, # of candidates per instance)
    original_scores = np.stack(scores)
    scores = original_scores.copy()
    if args.zero_mean:
        scores -= scores.mean(-1, keepdims=True)

    # Temporary dev/test split from dev set
    idxs = list(range(original_scores.shape[1]))
    np.random.shuffle(idxs)
    split_idx = int(len(idxs) * 0.8)

    train_scores = scores[:, idxs[:split_idx], :]
    test_original_scores = original_scores[:, idxs[split_idx:], :]
    test_scores = scores[:, idxs[split_idx:], :]

    logging.info("Fitting multivariate gaussian")
    train_scores_reshaped = train_scores.reshape(train_scores.shape[0], -1)
    mean = train_scores_reshaped.mean(1)
    cov = np.cov(train_scores_reshaped)

    logging.info("Running pruning prediction algorithm")
    total_score_random = 0
    total_score_baseline = 0
    total_score_pruning = 0
    runs_per_metric = [0] * test_scores.shape[0]
    for instance_idx in tqdm(range(test_scores.shape[1])):
        total_score_baseline += test_original_scores[-1, instance_idx].max()
        total_score_random += np.random.choice(test_original_scores[-1, instance_idx])

        instance_scores = test_scores[:, instance_idx]
        candidate_idxs = np.arange(train_scores.shape[2])

        for metric_idx in range(instance_scores.shape[0]-1):
            runs_per_metric[metric_idx] += instance_scores.shape[1]
            # print(f"METRIC {metric_idx+1}")
            known_values = instance_scores[:metric_idx+1].T
            y_given_x_mean, y_given_x_cov = conditional_mean_and_covar(known_values, mean, cov)
            estimated_best_idx = y_given_x_mean[:, -1].argmax()
            estimated_best_mean = y_given_x_mean[estimated_best_idx][-1]
            cdf = norm.cdf(y_given_x_mean[:, -1] - estimated_best_mean, scale=(y_given_x_cov[-1, -1] * 2)**0.5)
            instance_scores = instance_scores[:, cdf >= args.alpha]
            candidate_idxs = candidate_idxs[cdf >= args.alpha]

        runs_per_metric[-1] += candidate_idxs.size
        total_score_pruning += test_original_scores[-1, instance_idx, candidate_idxs].max()

    print(f"Final score -   random: {total_score_random / test_scores.shape[1]}")
    print(f"Final score - baseline: {total_score_baseline / test_scores.shape[1]}")
    print(f"Final score -  pruning: {total_score_pruning / test_scores.shape[1]}")
    for metric_name, num_runs in zip(model_names, runs_per_metric):
        print(f"Metric runs - {metric_name}: {num_runs / test_scores.shape[1]}")
    # import pdb; pdb.set_trace()



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

    args = parser.parse_args()
    main(args)
