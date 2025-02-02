import numpy as np
from scipy.stats import norm
from sklearn import linear_model
from collections import defaultdict
from read_data import get_train_test_data
import pandas as pd

MODEL_TYPE="quern" # riem, skintle, layers, quern
ZERO_MEAN = True
ALPHAS = [0.95, 0.9, 0.8, 0.7, 0.6]
DATA_PATH = "data.h5" #"data.h5"

def train_bayesian_models(train_metrics):
    num_columns = train_metrics.shape[1]
    bayesian_models = []
    for i in range(1, num_columns, 1):
        model = linear_model.BayesianRidge()
        X = train_metrics[:, :i]
        y = train_metrics[:, -1]
        bayesian_models.append(model.fit(X, y))
    return bayesian_models


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


def do_loop(test_metrics, bayesian_models, num_it=None, alpha=0.95):

    original_indices = np.arange(len(test_metrics))
    drop_dict = {}
    # iterate over test metrics (but not last "true" score)
    for i in range(len(test_metrics.T)-1): 

        # if len(test_metrics) <= 1: break

        mu_sigma = bayesian_models[i].predict(test_metrics[:,:i+1], return_std=True)
        mu_sigma = np.array(mu_sigma).T
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
    # Initialize a defaultdict to hold the sum of values for each key
    sum_dict = defaultdict(int)
    # Keep track of the count of dictionaries
    num_dicts = len(dicts_list)

    # Iterate over each dictionary in the list
    for d in dicts_list:
        for key, value in d.items():
            sum_dict[key] += value

    # Calculate the average for each key
    avg_dict = {key: value / num_dicts for key, value in sum_dict.items()}
    
    return avg_dict

print("Welcome :) ")

train_metrics, _ ,mu_train, sigma_train, final_test_matrix, _, _, orig_test_matrix = get_train_test_data(DATA_PATH, model_type=MODEL_TYPE, zero_mean=ZERO_MEAN)

bayesian_models = train_bayesian_models(train_metrics)
print("Training done! Starting inference...")
print()

drop_dicts = defaultdict(list)
best_comets = 0
winner_comets = defaultdict(float)
correct_ex = defaultdict(float)
not_dropped = defaultdict(float)

for t_i, test_metrics in enumerate(final_test_matrix):
    orig_test = orig_test_matrix[t_i]
    best = np.argmax(test_metrics.T[-1])
    best_comets += np.max(orig_test.T[-1])

    for a in ALPHAS:
        winner, drop_dict = do_loop(test_metrics, bayesian_models, alpha=a)
        winner_comets[a] += np.max(orig_test.T[-1][winner])
        #raise Exception
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
results_df.to_csv(f'bayesian_regression_results_{MODEL_TYPE}_zero_mean_{ZERO_MEAN}.csv', index=False, header=False)


    









