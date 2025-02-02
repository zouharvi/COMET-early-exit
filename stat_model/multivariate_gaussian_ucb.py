import numpy as np
from scipy.stats import norm
from read_data import get_train_test_data
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from collections import Counter


DATA_PATH = "sampling_models" #"new_models" # "data.h5"
MODEL_TYPE="skintle" # riem, skintle, layers, quern

USE_LOG_PROBS = True

if not USE_LOG_PROBS:
    METRIC_COSTS = [0.7  , 1.3  , 2.7  , 5.7]
    METRIC_CORRS = [0.133, 0.169, 0.188, 1.0]
else:
    METRIC_COSTS = [0, 0.7  , 1.3  , 2.7  , 5.7]
    METRIC_CORRS = [0, 0.133, 0.169, 0.188, 1.0]

CANDIDATE_SELECTION = "UCB" # "UCB" or "exp_value_max" # or random
COST_POWERS= [1] #[1, 2]
BETAS = [0.2] #[0.1, 0.2, 0.3 ]                  # only relevant for CANDIDATE_SELECTION = "UCB"
#BUDGET_CANDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
BUDGET_CANDS = [10, 50, 100]#, 150,  200]


#print(f"CANDIDATE_SELECTION = {CANDIDATE_SELECTION}\nCOST_POWER = {COST_POWER}\nBETA = {BETA}\nBUDGET_CAND = {BUDGET_CAND}")


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

    budget = num_ep * metric_costs[-1] - test_metrics.shape[0] * metric_costs[0]

    while budget > 0:

        if CANDIDATE_SELECTION == "UCB":
            UCB_candidate = get_UCB_candidate(mu_sigma, beta=beta)
        elif CANDIDATE_SELECTION == "exp_value_max":
            UCB_candidate = get_candidate(mu_sigma, test_metrics)
        elif CANDIDATE_SELECTION == "random":
            UCB_candidate = get_random_candidate(mu_sigma)
        else:
            raise NotImplementedError
        
        # UCB_metric = i+1
        UCB_metric = which_UCB_metric(mu, sigma, test_metrics[UCB_candidate, :].reshape(1,-1), observed_mask[UCB_candidate, :].reshape(1,-1), metric_costs=metric_costs, metric_corrs=metric_corrs, cost_power=cost_power)
        budget -= metric_costs[UCB_metric]

        # if we run out of budget, we can try cheaper metrics still
        while budget < 0 and UCB_metric > 1:
            budget += metric_costs[UCB_metric]
            UCB_metric -= 1
            budget -= metric_costs[UCB_metric]
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
    return winner, UCB_metrics

_, final_train_matrix ,_, _, _, _,_, orig_test_matrix = get_train_test_data(DATA_PATH, model_type=MODEL_TYPE, zero_mean=False, use_logprobs=USE_LOG_PROBS)

print("Done with loading the data, starting the meta model now!")
# get unique train examples
new_final_train = []
for instance in final_train_matrix:
    _, unique_idx =np.unique(instance, return_index=True, axis=0)
    instance = instance[unique_idx]

    # filter out candidates that are only 0s
    score_sums = np.sum(instance, axis=1)  
    non_zero_mask = score_sums != 0  
    instance = instance[non_zero_mask]
    ############## set log probs to random #################
    # instance[:,0] = np.random.random(instance[:,0].shape)
    new_final_train.append(instance)

reshaped_train_matrix = np.vstack(new_final_train)
mu_train = np.mean(reshaped_train_matrix, axis=0).reshape(-1, 1)
sigma_train = np.cov(reshaped_train_matrix, rowvar=False)

if CANDIDATE_SELECTION != "UCB": BETAS = [0] # if not UCB beta is not relevant

for BETA in BETAS:
    for COST_POWER in COST_POWERS:
       
        best_comets = defaultdict(float)
        winner_comets = defaultdict(float)
        correct_ex = defaultdict(float)
        random_comets = defaultdict(float)
        ucb_metrics_cand = dict()
        for BUDGET_CAND in BUDGET_CANDS:
            print("BETA", BETA, "COST POWER", COST_POWER, "BUDGET CAND", BUDGET_CAND)

            if not USE_LOG_PROBS:
                ucb_metrics_counts = {0:0, 1:0, 2:0, 3:0}
            else:
                ucb_metrics_counts = {0:0, 1:0, 2:0, 3:0, 4:0}
            num_cands = 0
            
            for t_i, orig_test in tqdm(enumerate(orig_test_matrix),total=len(orig_test_matrix)):

                # get unique test examples
                _ , unique_idx =np.unique(orig_test, return_index=True, axis=0)
                orig_test = orig_test[unique_idx]
                
                # filter out candidates that are only 0s
                score_sums = np.sum(orig_test, axis=1)  
                non_zero_mask = score_sums != 0  
                orig_test = orig_test[non_zero_mask]
            
                ############## set log probs to random #################
                # orig_test[:,0] = np.random.random(orig_test[:,0].shape)

                best = np.argmax(orig_test.T[-1])
                best_comets[BUDGET_CAND] += np.max(orig_test.T[-1])

                # the cheapest metric is already observed once for all to initialize
                ucb_metrics_counts[0] += orig_test.shape[0]
                num_cands += orig_test.shape[0]

                #num_ep=int(orig_test.shape[0]*BUDGET_CAND)
                num_ep = min(orig_test.shape[0], BUDGET_CAND)

                np.random.seed(42)
                random_cand = np.random.choice(orig_test.shape[0], max(1,num_ep), replace=False)

                random_comets[BUDGET_CAND] += np.max(orig_test[random_cand][:,-1])

                
                winner, ucb_metrics_list_i = do_loop(orig_test, mu_train, sigma_train, num_ep=num_ep, metric_costs=METRIC_COSTS, metric_corrs=METRIC_CORRS, cost_power=COST_POWER, beta=BETA)
                winner_comets[BUDGET_CAND] += orig_test.T[-1][winner]

                for m, count in Counter(ucb_metrics_list_i).items():
                    ucb_metrics_counts[m] += count

                correct = best == winner

                if correct:
                    correct_ex[BUDGET_CAND] += 1

            ucb_metrics_cand[BUDGET_CAND] = ucb_metrics_counts
        
        results_dict = defaultdict(list)


        for BUDGET_CAND in BUDGET_CANDS:
            num_samples = len(orig_test_matrix)

            total_cost = 0
            for key, value in ucb_metrics_cand[BUDGET_CAND].items():
                total_cost += METRIC_COSTS[key] * value/num_samples 
                results_dict[f"Metric {key+1} runs"].append(value/num_samples)
            results_dict["Correct Examples:"].append(correct_ex[BUDGET_CAND]/num_samples)
            results_dict["Model COMET score"].append(winner_comets[BUDGET_CAND]/num_samples)
            results_dict["Random COMET score"].append(random_comets[BUDGET_CAND]/num_samples)
            results_dict["Best COMET score"].append(best_comets[BUDGET_CAND]/num_samples)
             
            results_dict["Estimated cost"].append(total_cost)
            results_dict["Only last cost"].append(METRIC_COSTS[-1]*(num_cands/num_samples))

        # Create a DataFrame to store the results
        results_df = pd.DataFrame(results_dict).T.reset_index()
        results_df.to_csv(f'2abs_{MODEL_TYPE}_ucb_logprobs_{USE_LOG_PROBS}_candidate_sel_{CANDIDATE_SELECTION}_cost_power_{COST_POWER}_beta_{BETA}.csv', index=False, header=False)