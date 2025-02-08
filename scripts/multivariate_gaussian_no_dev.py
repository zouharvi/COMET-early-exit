import numpy as np
from scipy.stats import norm
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import argparse
from pathlib import Path
from lib.utils import average_dicts
import h5py
from lib import utils
import logging


def conditional_distr(test_data, num_observed_columns, test_confs=None, norm_confidences=False):
    norm_value = np.sqrt(np.pi/2) # go from absolute error to mean squared error
    std_devs = []
    means = []

    for index, x in enumerate(test_data): 
       
        x_1 = x[num_observed_columns-1:num_observed_columns] # (1, num_observed_columns)
        means.append(x_1)

        error_pred = test_confs[index][num_observed_columns-1:num_observed_columns]   # uncertainty is | score - target |

        if norm_confidences:
            error_pred = error_pred * norm_value

        x_1_var = np.diag(error_pred **2 ) 
        std_devs.append(x_1_var)

    means = np.array(means).reshape(-1, 1)
    std_devs = np.vstack(std_devs)


    return np.hstack([means, std_devs])

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

def do_loop(test_metrics, test_confs=None, alpha=0.95, norm_confidences=False):
    
    original_indices = np.arange(len(test_metrics))
    drop_dict = {}
    
    for i in range(len(test_metrics.T)-1): 
        mu_sigma = conditional_distr(test_metrics, i+1, test_confs=test_confs, norm_confidences=norm_confidences)
        
        drop_flags = calculate_confidences(mu_sigma, alpha=alpha)
        test_metrics = test_metrics[~drop_flags]
        drop_dict[i] = test_metrics.shape[0]
        original_indices = original_indices[~drop_flags]
    return original_indices, drop_dict


def read_data(args, model_class_name, use_confidences):



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
        if args.norm_confidences:
            output_path_base = work_dir / f"{args.model_class_name}_{args.generation_mode}_mvg_results_with_error_pred_norm_no_dev"
        else:
            output_path_base = work_dir / f"{args.model_class_name}_{args.generation_mode}_mvg_with_error_pred_no_dev"
    else:
        raise NotImplementedError
        output_path_base = work_dir / f"{args.model_class_name}_{args.generation_mode}_mvg_results"


    utils.configure_logger("multivariate_gaussian.py", output_path_base.with_suffix(".log"))
    logging.info("Reading data")
    train_scores, test_scores, test_confs = read_data(args, model_class_name=args.model_class_name, use_confidences=use_confidences)
    #_, baseline_test_scores, _ = read_data(args, model_class_name=args.baseline_model, use_confidences=use_confidences)

    logging.info("Prepairing train data")
    # train_scores = np.vstack(train_scores)
    # mu_train = np.mean(train_scores, axis=0).reshape(-1, 1)
    # sigma_train = np.cov(train_scores, rowvar=False)



    drop_dicts = defaultdict(list)
    best_comets = 0
    random_comets = 0
    winner_comets = defaultdict(float)
    winner_comets_baseline = defaultdict(float) 
    correct_ex = defaultdict(float)
    not_dropped = defaultdict(float)
    num_candidates = []

    random_subset = defaultdict(float)
    random_subset_winner = defaultdict(float)

    logging.info("Starting Gaussian model")

    for test_idx, test_score in tqdm(enumerate(test_scores), total=len(test_scores)):

        num_candidates.append(test_score.shape[0])
        best = np.argmax(test_score.T[-1])
        best_comets += np.max(test_score.T[-1])
        np.random.seed(42)
        random_cand = np.random.choice(test_score.shape[0], replace=False)
        random_comets += test_score[random_cand][-1]

        # get random subset comet
        subset_sizes = [round(v, 2) for v in list(np.arange(0.05, 1.05, 0.05))]
        for size in subset_sizes:

            subset_size = max(1, int(size * test_score.shape[0]))  # Compute subset size
            
            subset = np.random.choice(test_score.shape[0], size=subset_size, replace=False)  
            random_subset[size]+= max(test_score[subset][:,-1]) 
            correct_random = subset[np.argmax(test_score[subset][:,-1])] == best
            if correct_random:
                random_subset_winner[size] += 1 



        for a in args.alphas:
            test_conf = test_confs[test_idx] if test_confs is not None else None
            
            winner, drop_dict = do_loop(test_score, test_confs=test_conf, alpha=a, norm_confidences=args.norm_confidences)
            winner_comets[a] += np.max(test_score.T[-1][winner])
            #winner_comets_baseline[a] += np.max(baseline_test_scores[test_idx].T[-1][winner])
            drop_dicts[a].append(drop_dict)
            if len(winner) == test_score.shape[0]:
                not_dropped[a] +=1

            correct =  best in winner

            if correct:
                correct_ex[a] += 1


    num_samples = len(test_scores)
    for key in random_subset:
        random_subset[key] /= num_samples

    for key in random_subset_winner:
        random_subset_winner[key] /= num_samples   

    breakpoint()     

    import json
    with open(output_path_base.with_suffix(".json"), "w") as json_file:
        json.dump([dict(random_subset), dict(random_subset_winner)], json_file, indent=4)
    results_dict = defaultdict(list)

    
    for a in args.alphas:
        results_dict["Alpha"].append(a)
        results_dict["Pruned avg COMET score"].append(winner_comets[a]/num_samples)
        #results_dict["Pruned avg COMETKiwi score"].append(winner_comets_baseline[a]/num_samples)
        for key, value in average_dicts(drop_dicts[a]).items():
            results_dict[f"Avg. candidates left after observing score {key+1}"].append(value)
        results_dict["Correct Examples:"].append(correct_ex[a]/num_samples)
        results_dict["Random COMET score"].append(random_comets/num_samples)
        results_dict["Best COMET score"].append(best_comets/num_samples)

    # Create a DataFrame to store the results
    results_df = pd.DataFrame(results_dict).T.reset_index()
    results_df.to_csv(output_path_base.with_suffix(".csv"), index=False, header=False)
    logging.info("Finished")


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

    # parser.add_argument(
    #     "baseline_model",
    #     nargs="?",
    #     default="wmt22-cometkiwi-da",  # Set the default value
    #     help="Name of the model class, usually CometKiwi."
    # )

    parser.add_argument(
        "--use_confidences", action="store_true", help="Incorporate a model's error prediction.")
    
    parser.add_argument(
        "--norm_confidences", action="store_true", help="Normalize a model's error prediction.")

    parser.add_argument(
        "--alphas",
        type=float,
        nargs='+',
        default=[0.999, 0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.55, 0.52, 0.51, 0.505, 0.502, 0.501, 0.5005],
        help="List of alpha values (default: [0.95, 0.9, 0.8, 0.7, 0.6])"
    )

    parser.add_argument(
        "--generation_mode", type=str, default="sample", help="Either 'beam' or 'sample'.")



    args = parser.parse_args()
    main(args)


# python scripts/multivariate_gaussian.py vilem/scripts/data dev output wmt22-cometkiwi-da
# python scripts/multivariate_gaussian.py vilem/scripts/data dev output models-hydrogen
# python scripts/multivariate_gaussian.py vilem/scripts/data dev output models-lithium
# python scripts/multivariate_gaussian.py vilem/scripts/data dev output models-lithium --use_confidences
# python scripts/multivariate_gaussian.py vilem/scripts/data dev output models-lithium --use_confidences --norm_confidences
# python scripts/multivariate_gaussian.py vilem/scripts/data dev output models-beryllium
# python scripts/multivariate_gaussian.py vilem/scripts/data dev output models-beryllium --use_confidences
# python scripts/multivariate_gaussian.py vilem/scripts/data dev output models-beryllium --use_confidences --norm_confidences
# python scripts/multivariate_gaussian.py vilem/scripts/data dev output models-helium



# # python efficient_reranking/scripts/multivariate_gaussian.py vilem/scripts/data toy output wmt22-cometkiwi-da