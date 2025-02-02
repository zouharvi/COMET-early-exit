import numpy as np
from scipy.stats import norm
from sklearn import linear_model
from collections import defaultdict

ALPHAS = [0.95] #, 0.9, 0.8, 0.7, 0.6]
NUM_METRICS = 10
NUM_CANDIDATES_TEST = 10
NUM_TEST_MATRICES = 10000
TRAIN_SIZE = 1000000

def train_bayesian_models(train_metrics):
    num_samples = train_metrics.shape[0]
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

        # print(drop_flags)
        # print(len(np.where(drop_flags)[0].tolist()))
        # drop_dict[i] = len(np.where(drop_flags)[0].tolist())
        # print(test_metrics.shape)

        # Drop the rows marked as True in drop_flags from test_metrics
        test_metrics = test_metrics[~drop_flags]
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


def prepare_train(n):
    """
    Generate a random positive semi-definite covariance matrix with 1s on the diagonal, 
    and symmetric with off-diagonal values between 0 and 1.
    """
    # Create a random matrix A
    A = np.random.uniform(0, 1, size=(n, n))
    
    # Use Cholesky decomposition to generate a valid covariance matrix
    sigma_train = np.dot(A, A.T)
    
    # Normalize the matrix so that the diagonal is exactly 1
    D = np.sqrt(np.diag(sigma_train))
    sigma_train = sigma_train / np.outer(D, D)
    
    # Ensure diagonal elements are exactly 1
    np.fill_diagonal(sigma_train, 1)
    
    # Generate a random mean vector
    mu_train = np.random.uniform(0, 1, size=n).reshape(-1, 1)

    return sigma_train, mu_train

print("Welcome!")
sigma_train, mu_train = prepare_train(n=NUM_METRICS)
train_metrics = np.random.multivariate_normal(mu_train.flatten(), sigma_train, size=TRAIN_SIZE)
bayesian_models = train_bayesian_models(train_metrics)
print("Training done!")

num_samples = NUM_TEST_MATRICES


drop_dicts = defaultdict(list)
best_comets = 0
winner_comets = defaultdict(float)
correct_ex = defaultdict(float)
not_dropped = defaultdict(float)

for _ in range(num_samples):
    test_metrics = np.random.multivariate_normal(mu_train.flatten(), sigma_train, size=NUM_CANDIDATES_TEST)
    best = np.argmax(test_metrics.T[-1])
    best_comets += np.max(test_metrics.T[-1])

    for a in ALPHAS:
        winner, drop_dict = do_loop(test_metrics, bayesian_models, alpha=a)
        winner_comets[a] += np.max(test_metrics.T[-1][winner])
        drop_dicts[a].append(drop_dict)
        if len(winner) == NUM_CANDIDATES_TEST:
            not_dropped[a] +=1

        correct =  best in winner

        if correct:
            correct_ex[a] += 1


for a in ALPHAS:
    print("ALPHA", a)
    print("Correct Examples:", correct_ex[a]/num_samples)
    print("Baseline avg COMET Score", best_comets/num_samples)
    print("Pruned avg COMET score", winner_comets[a]/num_samples)
    print()
    for key, value in average_dicts(drop_dicts[a]).items():
        print(f"Avg. candidates left after observing score {key+1}: {value}")
    print("Nothing dropped:", not_dropped[a]/num_samples)
    print()