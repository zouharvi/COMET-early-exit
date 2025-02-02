import h5py
from collections import OrderedDict
import numpy as np
import os
PATH = "data.h5"


def read_h5_data(path, model_type=False, train_ratio=0.8):

    f = h5py.File(path, 'r')
    train_test_splits = OrderedDict()

    for k in f.keys():
        if k == "candidates" or k == 'scores_comet_wmt22-cometkiwi-da_0': 
            continue
        
        if model_type == "layers":
            if "cometkiwi" in k:
                data = f[k][:]
                split_index = int(train_ratio * len(data))
                train_test_splits[k] = {
                    "train": data[:split_index],
                    "test":  data[split_index:]
                }
        elif model_type == "quern":
            if not "cometkiwi" in k:
                data = f[k][:]
                split_index = int(train_ratio * len(data))
                train_test_splits[k] = {
                    "train": data[:split_index],
                    "test":  data[split_index:]
                }
        else:
            raise ValueError("Unknown model_type. Must be 'layers' or 'quern'.")
    
    f.close()
    if model_type == "layers":
        ordered_result = OrderedDict(sorted(train_test_splits.items(), key=lambda x: int(x[0].split("_")[-1])))
    else:
        predefined_order = ['scores_comet_quern-minilm', 'scores_comet_quern-bert-base-multilingual-cased', 
                            'scores_comet_quern-xlm-roberta-base', 'scores_comet_quern-xlm-roberta-large']

        ordered_result = OrderedDict(
            (key, value) for key, value in sorted(train_test_splits.items(), 
            key=lambda x: predefined_order.index(x[0])) 
        )

    final_train_matrix = np.stack([ordered_result[dataset_name]['train'] 
                      for dataset_name in ordered_result.keys()], axis=-1)

    final_test_matrix = np.stack([ordered_result[dataset_name]['test'] 
                     for dataset_name in ordered_result.keys()], axis=-1)
    return final_train_matrix, final_test_matrix


def read_folder_data(path, dropout=False, model_type="riem", train_ratio=0.8, similarity_path=None):
    if model_type not in ["skintle", "riem"]:
        raise ValueError
    
    if similarity_path and model_type!= "skintle":
        raise ValueError

    if similarity_path:
        sims = h5py.File(similarity_path)["similarities"]

    train_test_splits = OrderedDict() 
    
    for file_name in os.listdir(path):

        if file_name.endswith(".h5"):
            if model_type in file_name and ((dropout and "dropout" in file_name) or (not dropout and "dropout" not in file_name)) or "kiwi" in file_name:
                with h5py.File(os.path.join(path, file_name), 'r') as f:
                    data = f["scores"][:]
                if not dropout:
                    if "-S" in file_name:
                            train_test_splits['S'] = data
                    elif "-M" in file_name:
                        train_test_splits['M'] = data
                    elif "-L" in file_name:
                        train_test_splits['L'] = data
                    elif "cometkiwi-da" in file_name:  
                        train_test_splits['cometkiwi-da'] = data
                else:
                    train_test_splits.setdefault('S', []).append(data) if "-S" in file_name else None
                    train_test_splits.setdefault('M', []).append(data) if "-M" in file_name else None
                    train_test_splits.setdefault('L', []).append(data) if "-L" in file_name else None
                    train_test_splits.setdefault('cometkiwi-da', []).append(data) if "cometkiwi-da" in file_name else None


      
    ordered_result = OrderedDict((size, train_test_splits[size]) for size in ['S', 'M', 'L', 'cometkiwi-da'] if size in train_test_splits)

    if dropout:
        mean_dict = {}
        stdev_dict = {}

        for key, tensor_list in ordered_result.items():
            # Stack the 4 tensors along a new dimension (shape: (4, 7000, 128))
            stacked_tensors = np.stack(tensor_list, axis=0)

            score_sums = np.sum(stacked_tensors, axis=2)  
            non_zero_mask = np.any(score_sums != 0, axis=0)  
            filtered_stacked_tensors = stacked_tensors[:, non_zero_mask, :]  # Shape: (4, num_filtered_instances, 128)

            mean_dict[key] = np.mean(filtered_stacked_tensors, axis=0)    # Shape: (num_filtered_instances, 128)
            stdev_dict[key] = np.std(filtered_stacked_tensors, axis=0)   

        filtered_score_matrix = np.stack([mean_dict[dataset_name] 
                                            for dataset_name in mean_dict.keys()], axis=-1)  
        filtered_stdev_matrix = np.stack([stdev_dict[dataset_name] 
                                            for dataset_name in stdev_dict.keys()], axis=-1)     



    else:
        score_matrix = np.stack([ordered_result[dataset_name] 
                        for dataset_name in ordered_result.keys()], axis=-1)

        #breakpoint()
        # filter out zero scores
        score_sums = np.sum(score_matrix, axis=2)  
        non_zero_mask = score_sums != 0  
        instances_to_keep = np.any(non_zero_mask, axis=1)  
        filtered_score_matrix = score_matrix[instances_to_keep]
        if similarity_path:
            sims = sims[instances_to_keep]

    # split into train and test
    num_instances = filtered_score_matrix.shape[0]
    indices = list(range(num_instances))
    np.random.seed(42)  
    np.random.shuffle(indices)  

    split_index = int(num_instances * train_ratio) 

    train_data = filtered_score_matrix[indices[:split_index]] 
    test_data = filtered_score_matrix[indices[split_index:]]   

    if dropout:
        train_std_data = filtered_stdev_matrix[indices[:split_index]] 
        test_std_data = filtered_stdev_matrix[indices[split_index:]]

        return train_data, test_data, train_std_data, test_std_data, None, None
    
    if similarity_path:
        test_sim = sims[indices[split_index:]]
        train_sim = sims[indices[:split_index]] 

    else:
        test_sim, train_sim = None, None

    return train_data, test_data, None, None, train_sim, test_sim

def read_no_split_data(path, model_type="riem", similarity_path=None, use_logprobs=False):
    if model_type not in ["skintle", "riem"]:
        raise ValueError
    
    if similarity_path and model_type!= "skintle":
        raise ValueError

    if similarity_path:
        sims = h5py.File(similarity_path)["similarities"]

    train_test_splits = OrderedDict() 
    
    for file_name in os.listdir(path):

        if file_name.endswith(".h5"):
            if model_type in file_name or "kiwi" in file_name or "candidates" in file_name:
                
                with h5py.File(os.path.join(path, file_name), 'r') as f:
                    if "candidates" in file_name:
                        if use_logprobs:
                            data = f["token_logprobs"][:]
                            avg_log_probs = []
                            for instance in data:
                                candidates = []
                                for candidate in instance:
                                    candidate_mean = np.mean(candidate)
                                    # can happen if log probs are empty
                                    if np.isnan(candidate_mean): 
                                        candidate_mean = 0
                                    candidates.append(candidate_mean)
                                avg_log_probs.append(np.stack(candidates))

                            train_test_splits["log_probs"] = np.stack(avg_log_probs)
                        continue
                    else:
                        data = f["scores"][:]


                if "-S" in file_name:
                        train_test_splits['S'] = data
                elif "-M" in file_name:
                    train_test_splits['M'] = data
                elif "-L" in file_name:
                    train_test_splits['L'] = data
                elif "cometkiwi-da" in file_name:  
                    train_test_splits['cometkiwi-da'] = data

    ordered_result = OrderedDict((size, train_test_splits[size]) for size in ['log_probs', 'S', 'M', 'L', 'cometkiwi-da'] if size in train_test_splits)

    score_matrix = np.stack([ordered_result[dataset_name] 
                    for dataset_name in ordered_result.keys()], axis=-1)

    # filter out zero scores
    score_sums = np.sum(score_matrix, axis=2)  
    non_zero_mask = score_sums != 0  
    instances_to_keep = np.any(non_zero_mask, axis=1)  
    filtered_score_matrix = score_matrix[instances_to_keep]
    if similarity_path:
        sims = sims[instances_to_keep]
    else:
        sims = None
    return filtered_score_matrix, sims



def get_train_test_data(path, zero_mean=True, model_type="riem", dropout=False, similarity_path=None, use_logprobs=False):
    if model_type not in ["quern", "skintle", "riem", "layers"]:
        raise NotImplementedError(f"Unknown model type {model_type}")
    
    if path.endswith("h5"):
        if dropout:
            raise NotImplementedError("h5 file has no dropout scores!")
        final_train_matrix, final_test_matrix = read_h5_data(path, model_type=model_type)
        train_uncertainty, test_uncertainty, test_sim, train_sim = None, None, None, None
    elif "new_models" in path:
        final_train_matrix, final_test_matrix, train_uncertainty, test_uncertainty, train_sim, test_sim  = read_folder_data(path, model_type=model_type, 
                                                                                                       dropout=dropout, similarity_path=similarity_path)
        
    elif "sampling_models" in path:
        dev_path = path + "/" + "dev"
        test_path = path + "/" + "test_small"
        final_train_matrix, train_sim  = read_no_split_data(dev_path, model_type=model_type, similarity_path=similarity_path, use_logprobs=use_logprobs)
        final_test_matrix, test_sim  = read_no_split_data(test_path, model_type=model_type, similarity_path=similarity_path, use_logprobs=use_logprobs)
        train_uncertainty, test_uncertainty = None, None

    else:
        raise NotImplementedError
    if zero_mean:
        raise ValueError("Zero mean is no longer supported!")
        mean_across_scores = np.mean(final_train_matrix, axis=1, keepdims=True)
        zero_mean_matrix = final_train_matrix - mean_across_scores
        final_train_matrix = zero_mean_matrix

        mean_across_test_scores = np.mean(final_test_matrix, axis=1, keepdims=True)
        zero_mean_matrix_test = final_test_matrix - mean_across_test_scores

    else:
        zero_mean_matrix_test = final_test_matrix

    reshaped_train_matrix = final_train_matrix.reshape(-1, final_train_matrix.shape[-1])
    if train_uncertainty is not None:
        train_uncertainty = train_uncertainty.reshape(-1, train_uncertainty.shape[-1])

    mean_vector = np.mean(reshaped_train_matrix, axis=0).reshape(-1, 1)
    cov_matrix = np.cov(reshaped_train_matrix, rowvar=False)


    print("Model type:", model_type)
    print("Mean vector shape:", mean_vector.shape)
    print("Covariance matrix shape:", cov_matrix.shape)
    print("Final test matrix shape:", final_test_matrix.shape)  
    print("Final train matrix shape:", final_train_matrix.shape)
    print("Zero mean:", zero_mean)

    if similarity_path:
        return reshaped_train_matrix, final_train_matrix, mean_vector, cov_matrix, zero_mean_matrix_test, train_uncertainty, test_uncertainty, final_test_matrix, train_sim, test_sim
    return reshaped_train_matrix, final_train_matrix, mean_vector, cov_matrix, zero_mean_matrix_test, train_uncertainty, test_uncertainty, final_test_matrix

if __name__ == "__main__":
    ZERO_MEAN = True
    DATA_PATH = "new_models" #"new_models" # "data.h5"
    ALPHAS = [0.8] #[0.95, 0.9, 0.8, 0.7, 0.6]
    MODEL_TYPE="riem" # riem, skintle, layers, quern
    _, _ ,mu_train, sigma_train, final_test_matrix, _,_, _ = get_train_test_data(DATA_PATH, model_type=MODEL_TYPE, zero_mean=ZERO_MEAN)