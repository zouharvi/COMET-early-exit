import numpy as np
from scipy.stats import norm
from read_data import get_train_test_data
from collections import defaultdict
import pandas as pd
from tqdm import tqdm


ZERO_MEAN = False
DATA_PATH = "new_models" #"new_models" # "data.h5"
ALPHAS = [0.95, 0.9, 0.8, 0.7, 0.6]
MODEL_TYPE="skintle" # riem, skintle, layers, quern

def conditional_distr(mu, sigma, test_data, num_observed_columns):

    num_samples = test_data.shape[0]
    num_columns = test_data.shape[1]

    assert num_observed_columns <= num_columns, f"number of observed columns {num_observed_columns} cannot exceed the total number of columns {num_columns}"
    if num_observed_columns == 0:
        return np.stack([np.array([mu[-1,0], sigma[-1, -1]]) for i in range(num_samples)])
    if num_observed_columns == num_columns:
        # if we are at the last column, std is 0 because we can just pick
        return np.hstack([test_data[:,-1:], np.zeros_like(test_data[:,-1:])])

    mu_1  = mu[:num_observed_columns,:]
    mu_2  = mu[num_observed_columns:,:] 

    sigma_11 = sigma[:num_observed_columns, :num_observed_columns] # between observed variables
    sigma_12 = sigma[:num_observed_columns, num_observed_columns:]
    sigma_21 = sigma[num_observed_columns:, :num_observed_columns]
    sigma_22 = sigma[num_observed_columns:, num_observed_columns:] # between unobserved variables
    
    # calculate cond. distribution
    sigma_11_inv     = np.linalg.inv(sigma_11)
    sigma_22_given_x1 = sigma_22 - np.matmul(sigma_21, np.matmul(sigma_11_inv, sigma_12))

    stdev = np.sqrt(sigma_22_given_x1[-1,-1])

    mu_2_given_x1 = []
    for x in test_data:

        x_1 = x[:num_observed_columns]

        mu_2_given_x1_1 = mu_2 + np.matmul(sigma_21, np.matmul(sigma_11_inv, (x_1.reshape(-1,1) - mu_1)))
        mu_2_given_x1.append(mu_2_given_x1_1[-1,0])
    return np.hstack([np.array(mu_2_given_x1).reshape(-1,1), np.ones_like(test_data[:,-1:])*stdev])

def calculate_confidences(mu_sigma, alpha: float = 0.95):
    """
    find best_score, then (best_score - score / stdev) gives the confidence that score is worse than best_score. 
    Check whether this is greater than norm.ppf(alpha) (or whatever the chosen confidence bound).
    If greater, then drop the sample for further processing.
    """
    mu, sigma = mu_sigma[:,0], mu_sigma[0,1]
    mu_max = np.max(mu)
    confidence = (mu_max - mu) / (sigma)


    return confidence > norm.ppf(alpha) # if true drop it later

def do_loop(test_metrics, mu, sigma, num_it=None, alpha=0.95):
    
    original_indices = np.arange(len(test_metrics))
    drop_dict = {}

    # iterate over test metrics (but not last "true" score)
    for i in range(len(test_metrics.T)-1): 

        mu_sigma = conditional_distr(mu, sigma, test_metrics, i+1)
        
        drop_flags = calculate_confidences(mu_sigma, alpha=alpha)

        # Drop the rows marked as True in drop_flags from test_metrics
        test_metrics = test_metrics[~drop_flags]

        # store how big the test_metrix is after this step
        drop_dict[i] = test_metrics.shape[0]
        original_indices = original_indices[~drop_flags]

        if num_it:
            if i == num_it:
                break
    return original_indices, drop_dict

def average_dicts(dicts_list):
    sum_dict = defaultdict(int)
    num_dicts = len(dicts_list)
    # Iterate over each dictionary in the list
    for d in dicts_list:
        for key, value in d.items():
            sum_dict[key] += value

    # Calculate the average for each key
    avg_dict = {key: value / num_dicts for key, value in sum_dict.items()}
    
    return avg_dict


reshaped_train_matrix, final_train_matrix ,_, _, final_test_matrix, _,_, orig_test_matrix = get_train_test_data(DATA_PATH, model_type=MODEL_TYPE, zero_mean=ZERO_MEAN)

# get unique train examples
_, unique_idx =np.unique(reshaped_train_matrix, return_index=True, axis=0)
reshaped_train_matrix = reshaped_train_matrix[unique_idx]
mu_train = np.mean(reshaped_train_matrix, axis=0).reshape(-1, 1)
sigma_train = np.cov(reshaped_train_matrix, rowvar=False)


drop_dicts = defaultdict(list)
best_comets = 0
winner_comets = defaultdict(float)
correct_ex = defaultdict(float)
not_dropped = defaultdict(float)
num_candidates = []

for t_i, test_metrics in tqdm(enumerate(final_test_matrix),total=len(final_test_matrix)):
    orig_test = orig_test_matrix[t_i]

    # get unique test examples
    _ , unique_idx =np.unique(orig_test, return_index=True, axis=0)
    orig_test = orig_test[unique_idx]
    test_metrics = test_metrics[unique_idx]
    num_candidates.append(orig_test.shape[0])
    best = np.argmax(test_metrics.T[-1])
    best_comets += np.max(orig_test.T[-1])
    for a in ALPHAS:
        winner, drop_dict = do_loop(test_metrics, mu_train, sigma_train, num_it=None, alpha=a)
        winner_comets[a] += np.max(orig_test.T[-1][winner])
        drop_dicts[a].append(drop_dict)
        if len(winner) == test_metrics.shape[0]:
            not_dropped[a] +=1

        correct =  best in winner

        if correct:
            correct_ex[a] += 1

        
results_dict = defaultdict(list)


num_samples = len(final_test_matrix)
for a in ALPHAS:
    print("ALPHA", a)
    results_dict["Alpha"].append(a)
    print("Avg num of initial Candidates:", np.mean(num_candidates))
    print("Correct Examples:", correct_ex[a]/num_samples)
    print("Baseline avg COMET Score", best_comets/num_samples)
    print("Pruned avg COMET score", winner_comets[a]/num_samples)
    results_dict["Pruned avg COMET score"].append(winner_comets[a]/num_samples)
    print()
    for key, value in average_dicts(drop_dicts[a]).items():
        print(f"Avg. candidates left after observing score {key+1}: { value}")
        results_dict[f"Avg. candidates left after observing score {key+1}"].append(value)
    print("Nothing dropped:", not_dropped[a]/num_samples)
    results_dict["Correct Examples:"].append(correct_ex[a]/num_samples)
    print(10*"___")


# Create a DataFrame to store the results
results_df = pd.DataFrame(results_dict).T.reset_index()
results_df.to_csv(f'multivariate_gaussians_results_{MODEL_TYPE}_zero_mean_{ZERO_MEAN}.csv', index=False, header=False)