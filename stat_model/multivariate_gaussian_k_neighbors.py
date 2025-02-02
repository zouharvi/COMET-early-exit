import numpy as np
from scipy.stats import norm
from read_data import get_train_test_data
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import pandas as pd
np.random.seed(42)



ZERO_MEAN = True
NUM_NEIGHBOURS = [1000, 10000]
DATA_PATH = "new_models"
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
    confidence = (mu_max - mu) / sigma

    return confidence > norm.ppf(alpha) # if true drop it later


def get_nearest_train_distr(train_metrics, test_metrics, num_observed_columns, num_neighbors=200):
    train_metrics_selected = train_metrics[:,:,:num_observed_columns]  # shape becomes (num_inst, num_cand, num_cols )
    test_metrics_selected = test_metrics[:,:num_observed_columns]    # shape becomes (num_cand, num_cols )

    selected_indices = np.random.choice(train_metrics_selected.shape[1], 5, replace=False)
    train_metrics_sampled = train_metrics_selected[:, selected_indices, :]

    train_metrics_reshaped = train_metrics_sampled.reshape(-1, num_observed_columns)

    # NearestNeighbors model
    nn = NearestNeighbors(n_neighbors=num_neighbors, metric='euclidean')
    nn.fit(train_metrics_reshaped)
    distances, indices = nn.kneighbors(test_metrics_selected) 
    unique_indices_list = np.unique(indices.flatten()).tolist()

    train_metrics_full_sampled = train_metrics[:, selected_indices, :]
    train_metrics_reshaped_full = train_metrics_full_sampled.reshape(-1, train_metrics_full_sampled.shape[-1])

    nearest_train_metric = train_metrics_reshaped_full[unique_indices_list, :]

    # Compute the global mean and covariance on the zero-mean data
    mean_vector = np.mean(nearest_train_metric, axis=0).reshape(-1, 1)
    cov_matrix = np.cov(nearest_train_metric, rowvar=False)

    return mean_vector, cov_matrix


def do_loop(test_metrics, train_metrics_all, num_neighbours=200, alpha= 0.95, mu_trains=None, sigma_trains=None):

    original_indices = np.arange(len(test_metrics))
    drop_dict = {}

    mu_t = []
    sigma_t = []
    # iterate over test metrics (but not last "true" score)
    for i in range(len(test_metrics.T)-1): 
        if mu_trains == None:
            mu_train, sigma_train = get_nearest_train_distr(train_metrics=train_metrics_all, test_metrics=test_metrics, num_observed_columns=i+1, num_neighbors=num_neighbours)
            mu_t.append(mu_train)
            sigma_t.append(sigma_train)
        else:
            mu_train = mu_trains[i]
            sigma_train = sigma_trains[i]

        mu_sigma = conditional_distr(mu_train, sigma_train, test_metrics, i+1)
        

        drop_flags = calculate_confidences(mu_sigma, alpha=alpha)
        # Drop the rows marked as True in drop_flags from test_metrics
        test_metrics = test_metrics[~drop_flags]

        # store how big the test_metrix is after this step
        drop_dict[i] = test_metrics.shape[0]
        original_indices = original_indices[~drop_flags]

    return original_indices, drop_dict, mu_t, sigma_t


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


train_metrics_reshaped_orig, train_metrics, mu_train_all, sigma_train_all, final_test_matrix, _, _, orig_test_matrix = get_train_test_data(DATA_PATH, model_type=MODEL_TYPE, zero_mean=ZERO_MEAN, dropout=False)



for NUM_NEIGHBOR in NUM_NEIGHBOURS:
    print("NUM NEIGHBOR", NUM_NEIGHBOR)
    drop_dicts = defaultdict(list)
    best_comets = 0
    winner_comets = defaultdict(float)
    correct_ex = defaultdict(float)
    not_dropped = defaultdict(float)

    for t_i, test_metrics in tqdm(enumerate(final_test_matrix),total=len(final_test_matrix)):
        orig_test = orig_test_matrix[t_i]
        best = np.argmax(test_metrics.T[-1])
        best_comets += np.max(orig_test.T[-1])
        mu_trains = None
        sigma_trains = None
        for a_idx, a in enumerate(ALPHAS):

            if a_idx == 0:
                # store mu_trains and sigma_trains as they will be the same for the next alpha
                winner, drop_dict, mu_trains, sigma_trains = do_loop(test_metrics, train_metrics, num_neighbours = NUM_NEIGHBOR, alpha=a)
            else:

                winner, drop_dict, _, _ = do_loop(test_metrics, train_metrics, num_neighbours = NUM_NEIGHBOR, alpha=a, 
                                                                    mu_trains=mu_trains, sigma_trains=sigma_trains)
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
        # store things in dictionary to write it later to csv
        results_dict["Alpha"].append(a)
        results_dict["Pruned avg COMET score"].append(winner_comets[a]/num_samples)
        for key, value in average_dicts(drop_dicts[a]).items():
            results_dict[f"Avg. candidates dropped after observing score {key+1}"].append(value)
        results_dict["Correct Examples:"].append(correct_ex[a]/num_samples)

        # # also print outputs
        # print("ALPHA", a)
        # print("Correct Examples:", correct_ex[a]/num_samples)
        # print("Baseline avg COMET Score", best_comets/num_samples)
        # print("Pruned avg COMET score", winner_comets[a]/num_samples)
        # print()
        # for key, value in average_dicts(drop_dicts[a]).items():
        #     print(f"Avg. candidates dropped after observing score {key+1}: { value}")
        # print("Nothing dropped:", not_dropped[a]/num_samples)
        # print(10*"___")


    # Create a DataFrame to store the results
    results_df = pd.DataFrame(results_dict).T.reset_index()
    results_df.to_csv(f'kmeans_results_{MODEL_TYPE}_zero_mean_{ZERO_MEAN}_neighbours_{NUM_NEIGHBOR}.csv', index=False, header=False)





