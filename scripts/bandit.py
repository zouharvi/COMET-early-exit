import numpy as np
from scipy.stats import norm
from collections import defaultdict, Counter
import pandas as pd
from tqdm import tqdm
import argparse
from pathlib import Path
from lib.utils import average_dicts
import h5py
from lib import utils
import logging
import matplotlib.pyplot as plt
import copy
import pickle
import os





def read_data(args, use_confidences):

    h5_filename = Path(args.work_dir) / args.split / f"{utils.COMET_SCORES_H5DS_NAME}_comet_{args.model_class_name}_{args.generation_mode}.h5"

    h5_filename_candidates = Path(args.work_dir) / args.split / f"{utils.CANDIDATES_FILENAME}_{args.generation_mode}.h5"

    f = h5py.File(h5_filename, 'r')
    # sort layers and skip first one
    sorted_for_layers = sorted(f.keys(), key=lambda x: int(x.split("_")[-1]))[1:] # examples x candidates x layers

    data = np.stack([f[k][:] for k in sorted_for_layers], axis=-1)
    f.close()

    if args.logprobs:
        f_logprobs = h5py.File(h5_filename_candidates, 'r')
        data_logprobs = f_logprobs.get('token_logprobs')[()] # examples x candidates
        f_logprobs.close()


    if use_confidences:
        h5_filename_confidences = Path(args.work_dir) / args.split / f"{utils.COMET_CONFIDENCES_H5DS_NAME}_{args.model_class_name}_{args.generation_mode}.h5"
        f_conf = h5py.File(h5_filename_confidences, 'r')

        # sort layers and skip first one
        sorted_for_layers__conf = sorted(f_conf.keys(), key=lambda x: int(x.split("_")[-1]))[1:] # examples x candidates x layers

        data_conf = np.stack([f_conf[k][:] for k in sorted_for_layers__conf], axis=-1) 
        f.close()


    # filter out zero scores
    score_sums = np.sum(data, axis=2)
    non_zero_mask = score_sums != 0
    instances_to_keep = np.any(non_zero_mask, axis=1)
    data = data[instances_to_keep]


    if use_confidences:
        data_conf = data_conf[instances_to_keep]

    if args.logprobs:
        data_logprobs = data_logprobs[instances_to_keep]


    new_data = []; new_confs = []; new_logprobs = []
    for instance_idx, instance in enumerate(data):

        _, unique_idx =np.unique(instance, return_index=True, axis=0)
        instance = instance[unique_idx]
        score_sums = np.sum(instance, axis=1)
        non_zero_mask = score_sums != 0
        instance = instance[non_zero_mask]
        new_data.append(instance)
        if use_confidences:
            conf = data_conf[instance_idx]
            conf = conf[unique_idx]
            conf = conf[non_zero_mask]
            new_confs.append(conf)
        if args.logprobs:
            logprob = data_logprobs[instance_idx]
            logprob = logprob[unique_idx]
            logprob = logprob[non_zero_mask]
            new_logprobs.append(logprob)



    return new_data, new_confs, new_logprobs

def random_baseline(scores):
    random_subset = defaultdict(float)
    random_subset_winner = defaultdict(float)
    for test_score in scores:
        best = test_score[:,-1].max()
        np.random.seed(42)

        # get random subset comet
        subset_sizes = [round(v, 2) for v in list(np.arange(0.05, 1.05, 0.05))]
        for size in subset_sizes:

            subset_size = max(1, int(size * test_score.shape[0]))  # Compute subset size
            
            subset = np.random.choice(test_score.shape[0], size=subset_size, replace=False)  
            random_subset[size]+= max(test_score[subset][:,-1]) 
            correct_random = np.max(test_score[subset][:,-1]) == best
            if correct_random:
                random_subset_winner[size] += 1 
    num_samples = len(scores)
    for key in random_subset:
        random_subset[key] /= num_samples

    for key in random_subset_winner:
        random_subset_winner[key] /= num_samples  

    cost_random = sorted([float(k) for k in random_subset.keys()])  
    values_random = [random_subset[k] for k in cost_random] 

    cost_random_winners = sorted([float(k) for k in random_subset.keys()])  
    winners_random = [random_subset_winner[k] for k in cost_random_winners]
    return (cost_random, values_random), (cost_random_winners, winners_random)

def constant_baseline(scores):
    results = dict()
    n_layers = scores[0].shape[1]
    for i in range(n_layers):
        cand_scores_i = []
        best_scores_i = []
        for sample in scores:
            cand = sample[:,i].argmax()
            cand_score = sample[cand, -1]
            max_score = sample[:,-1].max()
            cand_eq_max = cand_score >= max_score
            cand_scores_i.append(cand_score)
            best_scores_i.append(cand_eq_max)
        results[(i+1)/n_layers] = {
            'avg_score': np.mean(cand_scores_i),
            'best_cand': np.mean(best_scores_i),
        }
    df_res = pd.DataFrame(results).T
    return df_res

def logprob_baseline(list_scores, list_logprobs, n_steps=20, mode='avg'):
    budgets = []
    avg_scores = []
    win_rates = []
    for i in range(1, n_steps+1, 1):
        budgets.append(i/n_steps)
        scores_i = []
        wins_i = []
        for scores, logprobs in zip(list_scores, list_logprobs):
            scores = scores[:,-1]
            max_score = scores.max()
            if mode == 'avg':
                avg_logprobs = np.array([l.mean() if l.size>0 else -np.inf for l in logprobs])
            elif mode == 'sum':
                avg_logprobs = np.array([l.sum() if l.size>0 else -np.inf for l in logprobs])
            else:
                avg_logprobs = None
                raise ValueError
            k = int(logprobs.shape[0] * (i/n_steps))
            arg_top_k_logprob = np.argpartition(avg_logprobs, -k)[-k:]
            top_k_scores = scores[arg_top_k_logprob]
            top_k_max_score = top_k_scores.max()
            win = top_k_max_score >= max_score
            scores_i.append(top_k_max_score)
            wins_i.append(win)
        avg_scores.append(np.mean(scores_i))
        win_rates.append(np.mean(wins_i))
    return budgets, avg_scores, win_rates
        


def bandit(config, scores, confs, start_layer=0, budget=0.5, ucb_factor=1.0):
    total_budget = scores.size * budget
    start_budget = scores.shape[0] * (start_layer+1)
    if start_budget > total_budget:
        return 0, 0

    assert total_budget >= start_budget, f'Total budget is less than the required budget to start with layer {start_layer}. Increase budget and/or decrease start_layer!'

    if config.norm_confidences:
        norm_factor = np.sqrt(np.pi / 2)
        confs = confs * norm_factor

    # Rescale Confidences by ucb_factor
    confs = confs * ucb_factor

    running_cost = start_budget
    max_layer_viewed = [start_layer] * scores.shape[0]
    # candidate_mask ensures candidates with all layers already viewed are ignored by setting to (-np.inf)
    candidate_mask = np.zeros(scores.shape[0])
    scores_est, confs_est = scores[:, start_layer], confs[:, start_layer]
    max_score = scores[:,-1].max()
    while running_cost < total_budget:
        # candidate_mask ensures candidates with all layers already viewed are ignored by setting to (-np.inf)
        ucb = scores_est + confs_est + candidate_mask
        if ucb.max() == -np.inf:
            break
        candidate = np.argmax(ucb)
        # view next layer
        max_layer_viewed[candidate] += 1
        if max_layer_viewed[candidate] >= scores.shape[1] - 1:
            candidate_mask[candidate] = -np.inf
        # Update candidate score and conf
        # print(candidate, candidate_mask[candidate], max_layer_viewed[candidate])
        scores_est[candidate] = scores[candidate, max_layer_viewed[candidate]]
        confs_est[candidate] = confs[candidate, max_layer_viewed[candidate]]
        # Update running cost
        running_cost += 1
    bandit_score = scores_est.max()
    bandit_win = bandit_score >= max_score
    return bandit_score, bandit_win

def run_bandits(config, scores_list, confs_list, **kwargs):
    scores_list = copy.deepcopy(scores_list)
    confs_list = copy.deepcopy(confs_list)
    bandit_scores = []
    bandit_wins = []
    for scores, confs in tqdm(zip(scores_list, confs_list), total=len(scores_list)):
        bandit_score, bandit_win = bandit(config, scores, confs, **kwargs)
        bandit_scores.append(bandit_score)
        bandit_wins.append(bandit_win)
    return np.mean(bandit_scores), np.mean(bandit_wins)


def bandit_new(config, scores, confs, start_layer=0, budgets=[0.5], ucb_factor=1.0):
    if not isinstance(budgets, list):
        budgets = [budgets]

    start_budget = scores.shape[0] * (start_layer+1)

    if config.norm_confidences:
        norm_factor = np.sqrt(np.pi / 2)
        confs = confs * norm_factor

    # Rescale Confidences by ucb_factor
    confs = confs * ucb_factor

    running_cost = start_budget
    max_layer_viewed = [start_layer] * scores.shape[0]
    # candidate_mask ensures candidates with all layers already viewed are ignored by setting to (-np.inf)
    candidate_mask = np.zeros(scores.shape[0])
    scores_est, confs_est = scores[:, start_layer], confs[:, start_layer]
    max_score = scores[:,-1].max()

    bandit_scores, bandit_wins = [], []
    max_layer_viewed_dict = {}
    for budget in budgets:
        total_budget = scores.size * budget
        if start_budget > total_budget:
            bandit_scores.append(0)
            bandit_wins.append(0)
            continue

        assert total_budget >= start_budget, f'Total budget is less than the required budget to start with layer {start_layer}. Increase budget and/or decrease start_layer!'

        while running_cost < total_budget:
            # candidate_mask ensures candidates with all layers already viewed are ignored by setting to (-np.inf)
            ucb = scores_est + confs_est + candidate_mask
            if ucb.max() == -np.inf:
                break
            candidate = np.argmax(ucb)
            # view next layer
            max_layer_viewed[candidate] += 1
            if max_layer_viewed[candidate] >= scores.shape[1] - 1:
                candidate_mask[candidate] = -np.inf
            # Update candidate score and conf
            # print(candidate, candidate_mask[candidate], max_layer_viewed[candidate])
            scores_est[candidate] = scores[candidate, max_layer_viewed[candidate]]
            confs_est[candidate] = confs[candidate, max_layer_viewed[candidate]]
            # Update running cost
            running_cost += 1
        bandit_score = scores_est.max()
        bandit_win = bandit_score >= max_score
        bandit_scores.append(bandit_score)
        bandit_wins.append(bandit_win)
        max_layer_viewed_dict[budget] = Counter(max_layer_viewed)
    return bandit_scores, bandit_wins, max_layer_viewed_dict

def run_bandits_new(config, scores_list, confs_list, **kwargs):
    scores_list = copy.deepcopy(scores_list)
    confs_list = copy.deepcopy(confs_list)
    bandit_scores = []
    bandit_wins = []
    max_layer_viewed_counts = None
    for scores, confs in tqdm(zip(scores_list, confs_list), total=len(scores_list)):
        bandit_score, bandit_win, max_layer_viewed = bandit_new(config, scores, confs, **kwargs)
        bandit_scores.append(bandit_score)
        bandit_wins.append(bandit_win)
        if max_layer_viewed_counts is None:
            max_layer_viewed_counts = max_layer_viewed
        else:
            for b in max_layer_viewed_counts:
                max_layer_viewed_counts[b] += max_layer_viewed[b]
    return np.array(bandit_scores).mean(axis=0), np.array(bandit_wins).mean(axis=0), max_layer_viewed_counts


def plot_things(results_dict, key_order=None):
    if key_order is None:
        key_order = list(results_dict.keys())
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 5))
    for key in key_order:
        value = results_dict[key]
        ax1.plot(*value['avg_score'], label=key)
        ax2.plot(*value['win_rate'], label=key)

    ax1.set_ylabel('Avg. Score', fontsize=12)
    ax1.set_xlabel('COMET layer (budget)', fontsize=12)
    random_scores = results_dict['random baseline']['avg_score'][1]
    ymin = np.floor(min(random_scores)) - 0.1
    ymax = np.ceil(max(random_scores)) + 0.1
    ax1.set_ylim((ymin, ymax))
    # ax1.grid()
    ax1.legend()

    ax2.set_ylabel('Win Rate', fontsize=12)
    ax2.set_xlabel('COMET layer (budget)', fontsize=12)
    # ax2.grid()
    ax2.legend()
    return fig

def save_things(results_dict, key_order=None):
    if key_order is None:
        key_order = list(results_dict.keys())
    data_tmp = {
        'scores': results_dict,
        'plot_order': key_order,
    }
    return data_tmp




def main(args):
    if args.use_confidences and args.model_class_name not in utils.CONFIDENCE_MODELS:
        raise ValueError(f"{args.model_class_name} does not support error prediction!")
    use_confidences = args.use_confidences and args.model_class_name in utils.CONFIDENCE_MODELS

    work_dir = Path(args.work_dir) / args.split
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path_base = work_dir / f"{args.model_class_name}_{args.generation_mode}_cst_results_no_dev"

    utils.configure_logger("constant_layer_pred.py", output_path_base.with_suffix(".log"))
    logging.info("Reading data")
<<<<<<< HEAD:scripts/bandit2.py
    scores, confs, logprobs = read_data(args, model_class_name=args.model_class_name, use_confidences=use_confidences)
=======
    scores, confs, logprobs = read_data(args, use_confidences=use_confidences)

    save_dir = f"figures/{args.split}/{args.model_class_name}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
>>>>>>> e8c73686f4e820b692f2e2adefe2b8b3ec43e9ee:scripts/bandit.py

    if args.subsamples is None:
        subset_size = len(scores)
    else:
        subset_size = args.subsamples
    scores, confs = scores[:subset_size], confs[:subset_size]

    # Add Baselines to Plots
    results_dict = dict()
    key_order = []
    # df_res = constant_baseline(scores)
    random_avg_tuple, random_winner_tuple = random_baseline(scores)
    results_dict['random baseline'] = {
        'avg_score': random_avg_tuple,
        'win_rate': random_winner_tuple,
    }
    key_order.append('random baseline')
    if args.logprobs:
        avg_logprob_budgets, avg_logprob_avg_scores, avg_logprob_win_rates = logprob_baseline(scores, logprobs, n_steps=20, mode='avg')
        sum_logprob_budgets, sum_logprob_avg_scores, sum_logprob_win_rates = logprob_baseline(scores, logprobs, n_steps=20, mode='sum')
        results_dict['top-k avg logprob'] = {
            'avg_score': (avg_logprob_budgets, avg_logprob_avg_scores),
            'win_rate': (avg_logprob_budgets, avg_logprob_win_rates),
        }
        key_order.append('top-k avg logprob')
        results_dict['top-k sum logprob'] = {
            'avg_score': (sum_logprob_budgets, sum_logprob_avg_scores),
            'win_rate': (sum_logprob_budgets, sum_logprob_win_rates),
        }
        key_order.append('top-k sum logprob')

    if args.start_layer_ablation:
        results_dict_start = copy.deepcopy(results_dict)
        key_order_start = copy.deepcopy(key_order)
        ucb_factor = 1.0
        # start_layers = [0,2,3,4,6,12]
        start_layers = [0,4,12]
        for start_layer in start_layers:
            print(f"start_layer = {start_layer}, ucb_factor = {ucb_factor}")
            bandit_budgets = np.arange(0.0,1.00001, 0.05)
            bandit_scores, bandit_win_rates, max_layer_viewed_counts = run_bandits_new(args, scores, confs, budgets=bandit_budgets.tolist(), start_layer=start_layer, ucb_factor=ucb_factor)
            # bandit_name = f'bandit (start={start_layer})'
            bandit_name = f'bandit ({start_layer}, {ucb_factor})'
            results_dict_start[bandit_name] = {
                'avg_score': (list(bandit_budgets), bandit_scores),
                'win_rate': (list(bandit_budgets), bandit_win_rates),
                'max_layers_viewed': max_layer_viewed_counts,
            }
            key_order_start.append(bandit_name)

        filename = f'{save_dir}/bandit_start_ablation_{str(subset_size)}'
        if args.norm_confidences:
            filename += '_norm'
        filename += f"_ucb{str(ucb_factor).replace('.', '')}"
        filename += str(args.generation_mode)

        fig = plot_things(results_dict_start, key_order=key_order_start)
        fig.savefig(filename)

        save_data = save_things(results_dict_start, key_order=key_order_start)
        with open(filename + '.pkl', 'wb') as fp:
            pickle.dump(save_data, fp)




    if args.ucb_ablation:
        results_dict_ucb = copy.deepcopy(results_dict)
        key_order_ucb = copy.deepcopy(key_order)
        start_layer = 0
        # ucb_factors = [0.0, 0.25, 0.5, 1.0, 2.0]
        ucb_factors = [0.0, 0.5, 1.0, 2.0]
        for ucb_factor in ucb_factors:
            print(f"start_layer = {start_layer}, ucb_factor = {ucb_factor}")
            bandit_budgets = np.arange(0.0,1.00001, 0.05)
            bandit_scores, bandit_win_rates, max_layer_viewed_counts = run_bandits_new(args, scores, confs, budgets=bandit_budgets.tolist(), start_layer=start_layer, ucb_factor=ucb_factor)
            # bandit_name = f'bandit (ucb_factor={ucb_factor})'
            bandit_name = f'bandit ({start_layer}, {ucb_factor})'
            results_dict_ucb[bandit_name] = {
                'avg_score': (list(bandit_budgets), bandit_scores),
                'win_rate': (list(bandit_budgets), bandit_win_rates),
                'max_layers_viewed': max_layer_viewed_counts,
            }
            key_order_ucb.append(bandit_name)

        filename = f'{save_dir}/bandit_ucb_ablation_{str(subset_size)}'
        if args.norm_confidences:
            filename += '_norm'
        filename += f'_start{str(start_layer)}'
        filename += str(args.generation_mode)

        fig = plot_things(results_dict_ucb, key_order=key_order_ucb)
        fig.savefig(filename)

        save_data = save_things(results_dict_ucb, key_order=key_order_ucb)
        with open(filename + '.pkl', 'wb') as fp:
            pickle.dump(save_data, fp)

    
    # Single Value Plot
    results_dict_single = copy.deepcopy(results_dict)
    key_order_single = copy.deepcopy(key_order)
    ucb_factor = 1.0
    start_layer = 0

    print(f"start_layer = {start_layer}, ucb_factor = {ucb_factor}")
    bandit_budgets = np.arange(0.0,1.00001, 0.05)
    bandit_scores, bandit_win_rates, max_layer_viewed_counts = run_bandits_new(args, scores, confs, budgets=bandit_budgets.tolist(), start_layer=start_layer, ucb_factor=ucb_factor)
    bandit_name = f'bandit ({start_layer}, {ucb_factor})'
    results_dict_single[bandit_name] = {
        'avg_score': (list(bandit_budgets), bandit_scores),
        'win_rate': (list(bandit_budgets), bandit_win_rates),
        'max_layers_viewed': max_layer_viewed_counts,
    }
    key_order_single.append(bandit_name)

<<<<<<< HEAD:scripts/bandit2.py
    filename = f"figures/test5/{args.model_class_name}_bandit_start_{str(subset_size)}_ucb_{str(ucb_factor).replace('.', '')}"
=======
    filename = f'{save_dir}/bandit_start_{str(subset_size)}_ucb_{str(ucb_factor).replace('.', '')}'
>>>>>>> e8c73686f4e820b692f2e2adefe2b8b3ec43e9ee:scripts/bandit.py
    if args.norm_confidences:
        filename += '_norm'
    filename += str(args.generation_mode)

    fig = plot_things(results_dict_single, key_order=key_order_single)
    fig.savefig(filename)

    save_data = save_things(results_dict_single, key_order=key_order_single)
    with open(filename + '.pkl', 'wb') as fp:
        pickle.dump(save_data, fp)



if __name__ == "__main__":
    #alphas = [round(x, 2) for x in list(np.arange(0.98, 0.6, -0.02))]

    alphas = [round(x, 2) for x in list(np.arange(0.98, 0.5, -0.05))]

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "data_dir", help="Data directory generated by the pipeline from vilem/scripts.")

    parser.add_argument(
        "split", type=str, help="Data split. Either 'dev' or 'test'.")

    parser.add_argument(
        "work_dir",  type=str, help="Working directory for all steps. "
                         "Will be created if doesn't exist.")
    
    parser.add_argument("model_class_name",
                        help="Name of the model class, e.g. 'hydrogen', 'lithium'." )
    parser.add_argument(
        "--use_confidences", action="store_true", help="Incorporate a model's error prediction.")
    
    parser.add_argument(
        "--norm_confidences", action="store_true", help="Normalize a model's error prediction.")
    
    parser.add_argument(
        "--ucb_ablation", action="store_true", help="Run UCB Ablation.")
    
    parser.add_argument(
        "--start_layer_ablation", action="store_true", help="Run Start Layer Ablation.")

    parser.add_argument(
        "--alphas",
        type=float,
        nargs='+',
        default=[0.999, 0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.55, 0.52, 0.51, 0.505, 0.502, 0.501, 0.5005],
        help="List of alpha values (default: [0.95, 0.9, 0.8, 0.7, 0.6])"
    )

    parser.add_argument(
        "--generation_mode", type=str, default="sample", help="Either 'beam' or 'sample'.")

    parser.add_argument(
        "--subsamples", type=int, default=None, help="Dataset size to run experiment on.")
    
    parser.add_argument(
        "--logprobs", action="store_true", help="Add logprob baselines.")



    args = parser.parse_args()
    main(args)



<<<<<<< HEAD:scripts/bandit2.py
# python scripts/bandit2.py vilem/scripts/data test_sample output_emb models-oxygen --use_confidences --norm_confidences --ucb_ablation --start_layer_ablation --subsamples 100

=======
# python scripts/bandit.py vilem/scripts/data test_sample   output models-oxygen --use_confidences --norm_confidences --generation_mode sample --start_layer_ablation --ucb_ablation --subsamples 100 --logprobs
>>>>>>> e8c73686f4e820b692f2e2adefe2b8b3ec43e9ee:scripts/bandit.py

# python scripts/bandit.py vilem/scripts/data dev           output models-oxygen --use_confidences --norm_confidences --generation_mode sample --start_layer_ablation --ucb_ablation --subsamples 100
