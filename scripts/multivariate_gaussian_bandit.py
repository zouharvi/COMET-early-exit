from collections import Counter
import numpy as np
from scipy.stats import norm
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import argparse
from pathlib import Path
import h5py
from lib import utils
import logging
from copy import deepcopy


def conditional_distr(mu, sigma, test_data, observed_mask):
    mu_2_given_x1 = []
    var_2_given_x1 = []
    for x, mask in zip(test_data, observed_mask):
        mu_1 = mu[mask,:]
        mu_2 = mu[~mask,:]
        x_1 = x[mask].reshape(-1,1)
        sigma_11 = sigma[mask][:,mask]
        sigma_12 = sigma[mask][:,~mask]
        sigma_21 = sigma[~mask][:,mask]
        sigma_22 = sigma[~mask][:,~mask]
    
        # calculate cond. distribution
        sigma_11_inv     = np.linalg.inv(sigma_11)
        sigma_22_given_x1 = sigma_22 - np.matmul(sigma_21, np.matmul(sigma_11_inv, sigma_12))
        
        mu_2_given_x1_1 = mu_2 + np.matmul(sigma_21, np.matmul(sigma_11_inv, (x_1.reshape(-1,1) - mu_1)))

        mu_2_given_x1_1_all = np.zeros_like(x)
        mu_2_given_x1_1_all[mask] = x[mask]
        mu_2_given_x1_1_all[~mask] = mu_2_given_x1_1.reshape(-1)

        var_2_given_x1_1_all = np.zeros_like(x)
        var_2_given_x1_1_all[~mask] = np.diagonal(sigma_22_given_x1).reshape(-1)

        var_2_given_x1.append(var_2_given_x1_1_all.reshape(1,-1))
        mu_2_given_x1.append(mu_2_given_x1_1_all.reshape(1,-1))

    return np.vstack(mu_2_given_x1), np.vstack(var_2_given_x1)

def calculate_UCB(mu_sigma, beta: float = 1):
    mu, sigma = mu_sigma[:,0], mu_sigma[0:,1]
    stdev = np.sqrt(sigma)
    return mu + beta * stdev

def get_UCB_candidate(mu_sigma, beta: float = 1):
    _, sigma = mu_sigma[:,0], mu_sigma[:,1]
    assert sigma.sum() > 0, "all last metrics observed"
    ucb = calculate_UCB(mu_sigma, beta)
    ucb_min = np.min(ucb) - 1
    ucb[sigma == 0] = ucb_min
    return np.argmax(ucb)

def get_random_candidate(mu_sigma):
    _, sigma = mu_sigma[:,0], mu_sigma[:,1]
    assert sigma.sum() > 0, "all last metrics observed"
    indices = np.where(sigma > 0)[0]
    # Randomly selecting one index where the value is greater than 0
    if len(indices) > 0:
        random_index = np.random.choice(indices)
    else:
        indices = np.where(sigma <= 0)[0]
        random_index = np.random.choice(indices)
    return random_index

def get_candidate(mu_sigma, test_metrics):
    mu, sigma = mu_sigma[:,0], mu_sigma[:,1]
    stdev = np.sqrt(sigma)
    assert sigma.sum() > 0, "all last metrics observed"
    mu_max = np.max(mu)
    epsilon = 0.000000000000001
    normalized_mu_max = (mu_max - mu )/ (stdev + epsilon)
    payoff = (norm.cdf(normalized_mu_max) * mu_max) + ((1 - norm.cdf(normalized_mu_max)) * mu) + (stdev * norm.pdf(normalized_mu_max)) 
    payoff[stdev == 0] = np.min(payoff) - 1

    ucb = calculate_UCB(mu_sigma)
    ucb_min = np.min(ucb) - 1
    ucb[sigma == 0] = ucb_min

    return np.argmax(payoff)

def which_UCB_metric(mu, sigma, test_data_x, observed_mask_x, metric_costs, metric_corrs, cost_power):
    assert observed_mask_x[0,-1] != True, "Last metric already observed!"
    prior_mu, prior_sigma = conditional_distr(mu, sigma, test_data_x, observed_mask_x)
    prior_mu_sigma = np.hstack([prior_mu[:,-1].reshape(-1,1), prior_sigma[:,-1].reshape(-1,1)])
    prior_std = np.sqrt(prior_mu_sigma[0,1])
    max_reward_margin = 0
    argmax_reward_margin = -1
    for m in range(1, observed_mask_x.shape[1], 1):
        if not observed_mask_x[0,m]:
            observed_mask_x[0,m] = True
            new_mu, new_sigma = conditional_distr(mu, sigma, test_data_x, observed_mask_x)
            new_mu_sigma = np.hstack([new_mu[:,-1].reshape(-1,1), new_sigma[:,-1].reshape(-1,1)])
            new_std = np.sqrt(new_mu_sigma[0,1])
            reward_margin = (prior_std - new_std) / metric_costs[m]**cost_power
            if reward_margin > max_reward_margin:
                max_reward_margin = reward_margin
                argmax_reward_margin = m
            observed_mask_x[0,m] = False
    return argmax_reward_margin

def do_loop(test_metrics, mu, sigma, num_ep, metric_costs, metric_corrs, cost_power, beta):
    
    observed_mask = np.zeros_like(test_metrics).astype(bool)
    observed_mask[:, :1] = 1

    mu2, sigma2 = conditional_distr(mu, sigma, test_metrics, observed_mask)
    mu_sigma = np.hstack([mu2[:,-1].reshape(-1,1), sigma2[:,-1].reshape(-1,1)])
    UCB_metrics = []

    original_budget = num_ep * metric_costs[-1] #- test_metrics.shape[0] * metric_costs[0]
    budget = deepcopy(original_budget)


    candidate_cost = {i: [[metric_costs[0]], [0]] for i in range(test_metrics.shape[0] + 1)}
    #candidate_cost = defaultdict(lambda: [[metric_costs[0]], [0]])

    #breakpoint()
    while budget > 0:

        if args.candidate_selection == "UCB":
            UCB_candidate = get_UCB_candidate(mu_sigma, beta=beta)
        elif  args.candidate_selection == "exp_value_max":
            UCB_candidate = get_candidate(mu_sigma, test_metrics)
        elif  args.candidate_selection == "random":
            UCB_candidate = get_random_candidate(mu_sigma)
        else:
            raise NotImplementedError
        
        # UCB_metric = i+1
        UCB_metric = which_UCB_metric(mu, sigma, test_metrics[UCB_candidate, :].reshape(1,-1), observed_mask[UCB_candidate, :].reshape(1,-1), metric_costs=metric_costs, metric_corrs=metric_corrs, cost_power=cost_power)
        # breakpoint()
        candidate_cost[UCB_candidate][0].append(metric_costs[UCB_metric])
        candidate_cost[UCB_candidate][1].append(UCB_metric)

        # sum costs so far but only take into account the last metric used
        # the cost for calculating the last metric also inlclude the cost for previous metrics
        budget = original_budget - sum(max(v[0]) for v in candidate_cost.values())

        # if we run out of budget, we can try cheaper metrics still
        while budget < 0 and UCB_metric > 1:

            # ignore previously added things
            candidate_cost[UCB_candidate][0] = candidate_cost[UCB_candidate][0][:-1]
            candidate_cost[UCB_candidate][1] = candidate_cost[UCB_candidate][1][:-1]

            # try next cheaper metric
            UCB_metric -= 1
            # if we already have calculated this metric, don't need to do again
            if UCB_metric == candidate_cost[UCB_candidate][1][-1]:
                break

            # add new metric and metric_cost and re-calculate budget
            candidate_cost[UCB_candidate][0].append(metric_costs[UCB_metric])
            candidate_cost[UCB_candidate][1].append(UCB_metric)
            budget = original_budget - sum(max(v[0]) for v in candidate_cost.values())
            
        # if even with the cheapest metric, we run out of budget, break
        if budget < 0:
            break


        observed_mask[UCB_candidate,UCB_metric] = True
        UCB_metrics.append(UCB_metric)

        mu2_x, sigma2_x = conditional_distr(mu, sigma, test_metrics[UCB_candidate, :].reshape(1,-1), observed_mask[UCB_candidate, :].reshape(1,-1))
        mu2[UCB_candidate,-1] = mu2_x[0,-1]
        sigma2[UCB_candidate,-1] = sigma2_x[0,-1]
        mu_sigma = np.hstack([mu2[:,-1].reshape(-1,1), sigma2[:,-1].reshape(-1,1)])

    
    winner =  np.argmax((mu2*observed_mask)[:,-1])

    breakpoint()
    return winner, UCB_metrics


def read_data(args, use_confidences):



    h5_filename = Path(args.work_dir) / args.split / f"{utils.COMET_SCORES_H5DS_NAME}_comet_{args.model_class_name}_{args.generation_mode}.h5"

    f = h5py.File(h5_filename, 'r')

    # sort layers and skip first one
    sorted_for_layers = sorted(f.keys(), key=lambda x: int(x.split("_")[-1]))[1:] # examples x candidates x layers

    data = np.stack([f[k][:] for k in sorted_for_layers], axis=-1) 
    f.close()

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

    new_data = []; new_confs = []
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


    # split into train and test
    num_instances = len(new_data)
    indices = list(range(num_instances))
    np.random.seed(42)  
    np.random.shuffle(indices)  
    train_ratio = 0.8
    split_index = int(num_instances * train_ratio) 

    train_data = [new_data[i] for i in indices[:split_index]]
    test_data = [new_data[i] for i in indices[split_index:]]

    if use_confidences:
        test_confs = [new_confs[i] for i in indices[split_index:]]
    else:
        test_confs = None
    return train_data, test_data, test_confs


def main(args):
    if args.use_confidences and args.model_class_name not in utils.CONFIDENCE_MODELS:
        raise ValueError(f"{args.model_class_name} does not support error prediction!")
    use_confidences = args.use_confidences and args.model_class_name in utils.CONFIDENCE_MODELS

    work_dir = Path(args.work_dir) / args.split
    work_dir.mkdir(parents=True, exist_ok=True)
    if use_confidences:
        output_path_base = work_dir / f"{args.model_class_name}_bandit_{args.candidate_selection}_with_confidences"
        raise NotImplementedError("Unfortunately, confidences can't be taken into account yet!")
    else:
        output_path_base = work_dir / f"{args.model_class_name}_bandit_{args.candidate_selection}"


    utils.configure_logger("multivariate_gaussian_bandit.py", output_path_base.with_suffix(".log"))
    logging.info("Reading data")
    train_scores, test_scores, test_confs = read_data(args, use_confidences=use_confidences)


    logging.info("Prepairing train data")
    train_scores = np.vstack(train_scores)
    mu_train = np.mean(train_scores, axis=0).reshape(-1, 1)
    sigma_train = np.cov(train_scores, rowvar=False)

    num_metrics = mu_train.shape[0]
    metric_costs =  [i / num_metrics for i in range(1, num_metrics + 1)]

    try:
        metric_corrs = utils.CORRELATIONS_WITH_LAST_LAYER[args.model_class_name]
    except KeyError:
        raise KeyError(f"{args.model_class_name} has no correlations added yet! :(")
    assert len(metric_corrs) == len(metric_costs)
    metric_corrs = metric_costs # for now set it to metric costs but actual corrs would be better
    if args.candidate_selection != "UCB": args.betas = [0] # if not UCB beta is not relevant

    for beta in args.betas:
        for cost_power in args.cost_powers:
        
            best_comets = defaultdict(float)
            winner_comets = defaultdict(float)
            correct_ex = defaultdict(float)
            random_comets = defaultdict(float)
            ucb_metrics_cand = dict()
            for budget_cand in args.budget_cands:
                logging.info(f"BETA: {beta}, COST POWER: {cost_power}, BUDGET CAND: {budget_cand}")

                ucb_metrics_counts = {i: 0 for i in range(num_metrics)}

                num_cands = 0
                
                for orig_test in tqdm(test_scores,total=len(test_scores)):


                    orig_test[:,0] = np.random.random(orig_test[:,0].shape)

                    best = np.argmax(orig_test.T[-1])
                    best_comets[budget_cand] += np.max(orig_test.T[-1])

                    # the cheapest metric is already observed once for all to initialize
                    ucb_metrics_counts[0] += orig_test.shape[0]
                    num_cands += orig_test.shape[0]
                    num_ep = min(orig_test.shape[0], budget_cand)

                    np.random.seed(42)

                    random_cand = np.random.choice(orig_test.shape[0], replace=False)
                    random_comets[budget_cand] += orig_test[random_cand][-1]


                    winner, ucb_metrics_list_i = do_loop(orig_test, mu_train, sigma_train, num_ep=num_ep, metric_costs=metric_costs, metric_corrs=metric_corrs, cost_power=cost_power, beta=beta)
                    breakpoint()
                    winner_comets[budget_cand] += orig_test.T[-1][winner]

                    for m, count in Counter(ucb_metrics_list_i).items():
                        ucb_metrics_counts[m] += count

                    if  best == winner:
                        correct_ex[budget_cand] += 1

                ucb_metrics_cand[budget_cand] = ucb_metrics_counts
            
            results_dict = defaultdict(list)


            for budget_cand in args.budget_cands:
                num_samples = len(test_scores)

                total_cost = 0
                for key, value in ucb_metrics_cand[budget_cand].items():
                    total_cost += metric_costs[key] * value/num_samples 
                    results_dict[f"Metric {key+1} runs"].append(value/num_samples)
                results_dict["Correct Examples:"].append(correct_ex[budget_cand]/num_samples)
                results_dict["Model COMET score"].append(winner_comets[budget_cand]/num_samples)
                results_dict["Random COMET score"].append(random_comets[budget_cand]/num_samples)
                results_dict["Best COMET score"].append(best_comets[budget_cand]/num_samples)
                
                results_dict["Estimated cost"].append(total_cost)
                results_dict["Only last cost"].append(metric_costs[-1]*(num_cands/num_samples))

            # Create a DataFrame to store the results
            results_df = pd.DataFrame(results_dict).T.reset_index()
            header = ["Candidate Budget"] + args.budget_cands
            results_df.to_csv(f'{output_path_base}_cost_power_{cost_power}_beta_{beta}.csv', index=False, header=header)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "data_dir", help="Data directory generated by the pipeline from vilem/scripts.")

    parser.add_argument(
        "split", type=str, help="Data split. Either 'dev' or 'test'.")

    parser.add_argument(
        "work_dir",  type=str, help="Working directory for all steps. "
                         "Will be created if doesn't exist.")
    
    parser.add_argument("model_class_name",
                        help="Name of the model class, e.g. 'quern', 'skintle'." )

    parser.add_argument(
        "--candidate_selection", 
        help="Type of candidate selection: UCB, exp_value_max or random",
        choices=["UCB", "exp_value_max", "random"], 
        default="UCB"
    )

    parser.add_argument(
        "--cost_powers", 
        type=float,
        nargs='+',
        default=[1], 
        help="List of cost powers (default: [1]). Will run experiments for all cost powers in this list"
    )

    parser.add_argument(
        "--use_confidences", action="store_true", help="Incorporate a models error prediction.")
    

    parser.add_argument(
        "--betas",
        type=float,
        nargs='+',
        default=[0.1], #, 0.2, 0.3 ],
        help="List of beta values (default: [0.95, 0.9, 0.8, 0.7, 0.6])"
    )

    parser.add_argument(
        "--budget_cands",
        type=int,
        nargs='+',
        default=[10,  50, 100, 150],
        help="List of beta values (default: [10,50, 100])"
    )

    parser.add_argument(
        "--generation_mode", type=str, default="sample", help="Either 'beam' or 'sample'.")

    args = parser.parse_args()
    main(args)

    # python scripts/multivariate_gaussian_bandit.py vilem/scripts/data dev output models-hydrogen
    # python scripts/multivariate_gaussian_bandit.py vilem/scripts/data dev output models-beryllium
    # python scripts/multivariate_gaussian_bandit.py vilem/scripts/data dev output models-hydrogen --candidate_selection exp_value_max
    # python scripts/multivariate_gaussian_bandit.py vilem/scripts/data dev output models-beryllium --candidate_selection exp_value_max