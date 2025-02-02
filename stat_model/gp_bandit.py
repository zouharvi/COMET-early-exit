import numpy as np
from read_data import get_train_test_data
from scipy.stats import pearsonr, norm, kendalltau

DATA_PATH = "new_models" #"new_models" # "data.h5"
MODEL_TYPE="skintle" # riem, skintle, layers, quern
SIMILARITY_PATH = "similarities_skintle-S.h5"

from tqdm import tqdm


def prepare_data(data_path, model_type, sim_path):
    _, final_train_matrix ,mu_train, sigma_train, final_test_matrix, _,_ , orig_test, train_sim, test_sim = get_train_test_data(data_path, model_type=model_type, zero_mean=False, similarity_path=sim_path)
    mean = mu_train.reshape(-1)
    cov = sigma_train
    test_scores = np.transpose(final_test_matrix, (2, 0, 1))
    test_original_scores = np.transpose(orig_test, (2, 0, 1))
    train_scores = np.transpose(final_train_matrix, (2, 0, 1))
    return mean, cov, test_scores, test_original_scores, train_scores, train_sim, test_sim

def kernel(sim):
    RBF_SHAPE = .8
    return np.exp(-(1 - sim) / (2 * RBF_SHAPE ** 2))

def run_bandit(test_original_scores, test_sims):

    sim_h5ds = test_sims
    correlations = []
    chosen_is_best = []
    chosen_best_candidate_scores = []
    best_known_candidate_scores = []

    random_scores = []

    best_candidates_scores = []
    for i in tqdm(range(test_original_scores.shape[1]), total=test_original_scores.shape[1]):
        scores = test_original_scores[:, i, :]
        sims = sim_h5ds[i]

        # remove duplicate candidates and their corresponding sim
        _, unique_indices = np.unique(sims, return_index=True, axis=1)
        unique_indices = sorted(unique_indices)
        scores = scores[:,unique_indices]
        sims = sims[unique_indices][:, unique_indices]

        # # make the scores zero mean
        # scores -=  np.mean(scores, axis=1, keepdims=True)

        # only use best (=last) metric
        scores = scores[-1:,].squeeze()

        # best_scores
        best_candidate =   np.argmax(scores)
        best_score = np.max(scores)
        random_scores.append(np.random.choice(scores).item())

        INITIAL_SIZE = int(scores.shape[0]/4)
        all_idxs = np.arange(scores.shape[0])
        np.random.seed(42)
        known_idxs = np.random.choice(scores.shape[0], INITIAL_SIZE, replace=False)
        unknown_idxs = np.array([x for x in all_idxs if x not in known_idxs])
        # rbf_cov = np.zeros((unknown_idxs.size, unknown_idxs.size))
        prior_var = np.var(scores[known_idxs])

        rbf_cov = np.zeros((all_idxs.size, all_idxs.size))

        for j in all_idxs:
            for k in all_idxs:
                rbf_cov[j, k] = kernel(sim=sims[k, j])

        unknown_unknown_cov = rbf_cov[unknown_idxs][:, unknown_idxs]
        known_unknown_cov = rbf_cov[known_idxs][:, unknown_idxs]
        known_known_cov = rbf_cov[known_idxs][:, known_idxs]

 
        prior_cov = 0
        inverse_known_known_plus_prior = np.linalg.inv(known_known_cov + prior_cov)

        term_1 = np.matmul(inverse_known_known_plus_prior, known_unknown_cov)
        posterior_cov = unknown_unknown_cov - np.matmul(known_unknown_cov.T, term_1)

        mean_term_1 = np.matmul(inverse_known_known_plus_prior, scores[known_idxs])
        posterior_mean = np.matmul(known_unknown_cov.T, mean_term_1)

        posterior_mean_all = np.zeros_like(scores)
        posterior_mean_all[known_idxs] = scores[known_idxs]
        posterior_mean_all[unknown_idxs] = posterior_mean
        
        posterior_var_all = np.zeros_like(scores)

        chosen_candidate = np.argmax(posterior_mean_all)
        chosen_candidate_score = scores[chosen_candidate]
       
        chosen_best_candidate_scores.append(chosen_candidate_score)
        best_candidates_scores.append(best_score)
        chosen_is_best.append(best_candidate == chosen_candidate)
        best_known_candidate_scores.append(scores[known_idxs].max())

        correlations.append(pearsonr(posterior_mean, scores[unknown_idxs])[0])
        #print(pearsonr(posterior_mean, scores[unknown_idxs]))

    print()
    print("AVERAGE CORR", np.mean(correlations))
    print("BEST CANDIDATES SCORE", np.mean(best_candidates_scores))
    print("BEST KNOWN CANDIDATES SCORE", np.mean(best_known_candidate_scores))
    print("BEST CHOSEN CANDIDATES SCORE", np.mean(chosen_best_candidate_scores))
    print("CHOSEN IS BEST", np.mean(chosen_is_best))
    print("RANDOM BASELINE", np.mean(random_scores))




def main(data_path, model_type, similarity_path):
    mean, cov, test_scores, test_original_scores, train_scores, train_sim, test_sim = prepare_data(data_path, model_type, sim_path=similarity_path)
    run_bandit(test_original_scores, test_sim)
if __name__ == "__main__":
    main(DATA_PATH, MODEL_TYPE, SIMILARITY_PATH)