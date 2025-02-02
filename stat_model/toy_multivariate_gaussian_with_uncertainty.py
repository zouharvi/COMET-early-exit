import numpy as np
from scipy.stats import norm
np.random.seed(31)
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

def do_loop(test_metrics, test_std_devs, mu, sigma, num_it=None):

    original_indices = np.arange(len(test_metrics))
    # iterate over test metrics (but not last "true" score)
    for i in range(len(test_metrics.T)-1): 

        # if len(test_metrics) <= 1: break

        mu_sigma = conditional_distr(mu, sigma, test_metrics, test_std_devs,  i+1)
        drop_flags = calculate_confidences(mu_sigma)

        # Drop the rows marked as True in drop_flags from test_metrics
        test_metrics = test_metrics[~drop_flags]
        test_std_devs = test_std_devs[~drop_flags]
        original_indices = original_indices[~drop_flags]
        if num_it:
            if i == num_it:
                break
    return original_indices

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



NUM_METRICS = 10
NUM_SAMPLES_TEST = 5
TEST_MATRICES = 10000

sigma_train, mu_train = prepare_train(n=NUM_METRICS)

correct_ex = 0
not_dropped = 0
num_samples = TEST_MATRICES
for _ in range(num_samples):

    
    test_metrics = np.random.multivariate_normal(mu_train.flatten(), sigma_train, size=NUM_SAMPLES_TEST)

    # Calculate the standard deviation (10% of the mean)
    percentage_error = 0.1  # 10% error
    std_devs_mean = percentage_error * mu_train.flatten()  # Shape (NUM_METRICS,)
    test_std_devs = np.abs(np.random.normal(loc=std_devs_mean, scale=0.5, size=(NUM_SAMPLES_TEST, NUM_METRICS)))
    
    best = np.argmax(test_metrics.T[-1])
    breakpoint()
    winner = do_loop(test_metrics, test_std_devs, mu_train, sigma_train)
    if len(winner) == NUM_SAMPLES_TEST:
        not_dropped +=1

    correct =  best in winner

    if correct:
        correct_ex += 1
        
print("Correct Examples:", correct_ex/num_samples)
print("Nothing dropped:", not_dropped/num_samples)