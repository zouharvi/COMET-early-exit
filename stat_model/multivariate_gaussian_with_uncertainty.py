import numpy as np
from scipy.stats import norm
np.random.seed(31)
from read_data import get_train_test_data
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

ZERO_MEAN = True
DATA_PATH = "new_models" #"new_models" # "data.h5"
ALPHAS = [0.95, 0.9, 0.8, 0.7, 0.6]
MODEL_TYPE="riem" # riem, skintle, layers, quern

def conditional_distr(mu, sigma, test_data, test_std_devs,  num_observed_columns):

    num_samples = test_data.shape[0]
    num_columns = test_data.shape[1]
    
    assert num_observed_columns <= num_columns, f"number of observed columns {num_observed_columns} cannot exceed the total number of columns {num_columns}"
    if num_observed_columns == 0:
        return np.stack([np.array([mu[-1,0], sigma[-1, -1]]) for i in range(num_samples)])
    if num_observed_columns == num_columns:
        return np.hstack([test_data[:,-1:], np.zeros_like(test_data[:,-1:])])
    
    mu_1  = mu[:num_observed_columns,:]
    mu_2  = mu[num_observed_columns:,:]  
    
    sigma_11 = sigma[:num_observed_columns, :num_observed_columns] # between observed variables
    sigma_12 = sigma[:num_observed_columns, num_observed_columns:]
    sigma_21 = sigma[num_observed_columns:, :num_observed_columns]
    sigma_22 = sigma[num_observed_columns:, num_observed_columns:] # between unobserved variables
    
    # calculate cond. distribution
    sigma_11_inv     = np.linalg.inv(sigma_11) # (1, 1)
    sigma_22_given_x1_all = sigma_22 - np.matmul(sigma_21, np.matmul(sigma_11_inv, sigma_12)) 

   
    
    mu_2_given_x1 = []
    std_devs = []
    for index, x in enumerate(test_data): 
        x_1 = x[:num_observed_columns] # (1, num_observed_columns)
        mu_2_given_x1_1 = mu_2 + np.matmul(sigma_21, np.matmul(sigma_11_inv, (x_1.reshape(-1,1) - mu_1)))
        mu_2_given_x1.append(mu_2_given_x1_1[-1,0])

        # here is the uncertainty for each score added to the Cov
        x_1_var = np.diag(test_std_devs[index][:num_observed_columns] **2 ) # assume that uncertainty comes as std
        sigma_22_given_x1 = sigma_22_given_x1_all  + np.matmul(np.matmul(sigma_21, sigma_11_inv), np.matmul(x_1_var, np.matmul(sigma_11_inv, sigma_12)))
        stdev = np.sqrt(sigma_22_given_x1[-1,-1]) 
        std_devs.append(stdev)

    std_devs = np.vstack(std_devs)

    return np.hstack([np.array(mu_2_given_x1).reshape(-1,1), std_devs]) #5,1

def calculate_confidences(mu_sigma, alpha: float = 0.95):
    """
    find best_score, then (best_score - score / stdev) gives the confidence that score is worse than best_score. 
    Check whether this is greater than norm.ppf(alpha) (or whatever the chosen confidence bound).
    If greater, then drop the sample for further processing.
    """
    mu, sigma = mu_sigma[:,0], mu_sigma[0,1]
    mu_max = np.max(mu)
    confidence = (mu_max - mu) / sigma
    return confidence > norm.ppf(alpha) # if true drop it later

def do_loop(test_metrics, test_std_devs, mu, sigma, num_it=None, alpha=0.95):

    original_indices = np.arange(len(test_metrics))
    drop_dict = {}
    # iterate over test metrics (but not last "true" score)
    for i in range(len(test_metrics.T)-1): 

        mu_sigma = conditional_distr(mu, sigma, test_metrics, test_std_devs,  i+1)
        drop_flags = calculate_confidences(mu_sigma, alpha=alpha)

        # Drop the rows marked as True in drop_flags from test_metrics
        test_metrics = test_metrics[~drop_flags]
        test_std_devs = test_std_devs[~drop_flags]
        drop_dict[i] = test_metrics.shape[0]
        original_indices = original_indices[~drop_flags]

    return original_indices,  drop_dict


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


train_metrics, _ ,mu_train, sigma_train, final_test_matrix, train_uncertainty, test_uncertainty, orig_test_matrix = get_train_test_data(DATA_PATH, model_type=MODEL_TYPE,
                                                                                                                      zero_mean=ZERO_MEAN, dropout=True)


drop_dicts = defaultdict(list)
best_comets = 0
winner_comets = defaultdict(float)
correct_ex = defaultdict(float)
not_dropped = defaultdict(float)

for t_idx, test_metrics in tqdm(enumerate(final_test_matrix), total=len(final_test_matrix)):
    orig_test = orig_test_matrix[t_idx]
    best = np.argmax(test_metrics.T[-1])
    best_comets += np.max(orig_test.T[-1])
    for a in ALPHAS:

        winner, drop_dict = do_loop(test_metrics, test_uncertainty[t_idx], mu_train, sigma_train,alpha=a)
        
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
    print("Correct Examples:", correct_ex[a]/num_samples)
    print("Baseline avg COMET Score", best_comets/num_samples)
    print("Pruned avg COMET score", winner_comets[a]/num_samples)
    results_dict["Pruned avg COMET score"].append(winner_comets[a]/num_samples)
    print()
    for key, value in average_dicts(drop_dicts[a]).items():
        print(f"Avg. candidates dropped after observing score {key+1}: { value}")
        results_dict[f"Avg. candidates dropped after observing score {key+1}"].append(value)
    print("Nothing dropped:", not_dropped[a]/num_samples)
    results_dict["Correct Examples:"].append(correct_ex[a]/num_samples)
    print(10*"___")


# Create a DataFrame to store the results
results_df = pd.DataFrame(results_dict).T.reset_index()
results_df.to_csv(f'uncertainty_multivariate_gaussians_results_{MODEL_TYPE}_zero_mean_{ZERO_MEAN}.csv', index=False, header=False)




