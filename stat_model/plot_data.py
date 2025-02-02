import h5py
from collections import OrderedDict
import numpy as np
PATH = "data.h5"

import matplotlib.pyplot as plt
import seaborn as sns

def custom_order_key(name):
    # Define the order you want
    order = {
        "mini": 0,
        "bert": 1,
        "roberta-base": 2,
        "roberta-large": 3
    }
    
    # Check which keyword is in the string and return the corresponding order
    if "mini" in name:
        return order["mini"]
    elif "bert" in name:
        return order["bert"]
    elif "roberta-base" in name:
        return order["roberta-base"]
    elif "roberta-large" in name:
        return order["roberta-large"]
    else:
        raise ValueError

def get_train_test_data(path, layers=False, zero_mean=True):

    f = h5py.File(path)
    # Show all datasets

    ordered_scores = []

    for k in f.keys():
        if k == "candidates": 
            continue
        if k == 'scores_comet_wmt22-cometkiwi-da_0': 
            continue
        if layers:
            if "cometkiwi" in k:
                ordered_scores.append(k)
        else:
            if not "cometkiwi" in k:
                ordered_scores.append(k)
            
    if layers:
        ordered_scores = sorted(ordered_scores, key=lambda x:int(x.split("_")[-1]))

    else:
        ordered_scores = sorted(ordered_scores, key=custom_order_key)

    train_test_splits = OrderedDict()

    # Iterate over the ordered datasets
    for dataset_name in ordered_scores:
        data = f[dataset_name][:]
 
        # Determine the split index (80% train, 20% test)
        split_index = int(0.8 * len(data))

        # Split the data
        train_data = data[:split_index]
        test_data = data[split_index:]
        
        # Store the split in the dictionary or process accordingly
        train_test_splits[dataset_name] = {
            "train": train_data,
            "test": test_data
        }

    f.close()

    train_matrices = []

    # Ensure we only concatenate the train data from each dataset
    for dataset_name in train_test_splits.keys():
        train_matrices.append(train_test_splits[dataset_name]['train'])

    # Stack the train data along the last axis (axis=-1)
    # This will give the final shape (2403, 256, 25)
    final_train_matrix = np.stack(train_matrices, axis=-1)

    if zero_mean:
        mean_across_scores = np.mean(final_train_matrix, axis=1, keepdims=True)

        # Subtract the mean from each element along the last dimension
        zero_mean_matrix = final_train_matrix - mean_across_scores
        final_train_matrix = zero_mean_matrix

    # Reshape the matrix to flatten the first two dimensions (2403, 256) into one dimension
    reshaped_train_matrix = final_train_matrix.reshape(-1, final_train_matrix.shape[-1])


    # Compute the global mean and covariance on the zero-mean data
    mean_vector = np.mean(reshaped_train_matrix, axis=0)
    cov_matrix = np.cov(reshaped_train_matrix, rowvar=False)

    # Reshape mean_vector 
    mean_vector = mean_vector.reshape(-1, 1)

    # Print the final shapes to verify
    print("Mean vector shape:", mean_vector.shape)
    print("Covariance matrix shape:", cov_matrix.shape)

    test_matrices = []

    # Iterate over the ordered datasets and collect the test data
    for dataset_name in train_test_splits.keys():
        # Append the test data (shape should be (test_len, )) for each dataset

        test_matrices.append(train_test_splits[dataset_name]['test'])

    # Stack the test data along the last axis (axis=-1)
    final_test_matrix = np.stack(test_matrices, axis=-1)

    if zero_mean:
        mean_across_scores = np.mean(final_test_matrix, axis=0, keepdims=True)

        # Subtract the mean from each element along the last dimension
        zero_mean_matrix = final_test_matrix - mean_across_scores
        final_test_matrix = zero_mean_matrix

    # Print the final shape to verify
    print("Final test matrix shape:", final_test_matrix.shape)  
    print("Zero mean:", zero_mean)

    #raise Exception


    return reshaped_train_matrix, mean_vector, cov_matrix, final_test_matrix


def plot_score_distribution(data, num_cols=6, plot_name="data_plot.png"):
    num_score_types = data.shape[2]
    num_rows = (num_score_types + num_cols - 1) // num_cols  # Calculate the number of rows needed

    # Find the global min and max of all the scores to set consistent x-axis limits
    global_min = np.min(data)
    global_max = np.max(data)

    # Create figure with a grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 4 * num_rows))

    # Flatten axes to iterate through them easily
    axes = axes.flatten()

    for i in range(num_score_types):
        # Collect all scores for the i-th score type across all instances and candidates
        score_type_data = data[:, :, i].flatten()  # Flatten the data for this score type
        
        # Plot normalized histogram (density=True for normalized histogram)
        axes[i].hist(score_type_data, bins=1000, alpha=0.7, color='blue', density=True)
        axes[i].set_title(f'Model {i + 1}')
        axes[i].set_xlabel('Score')
        axes[i].set_ylabel('Density')

        # Set consistent x-axis limits across all plots
        axes[i].set_xlim(global_min, global_max)

    # Remove empty subplots if num_score_types is not a perfect multiple of num_cols
    for j in range(num_score_types, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout to prevent label overlap
    plt.tight_layout()

    # Save the figure
    plt.savefig(plot_name)


def plot_scatter_score_vs_final(data, num_cols=6, plot_name= 'scatter_score_vs_final_grid.png', layers=True, zero_mean=True):
    num_score_types = data.shape[2] - 1  # Exclude the final score type
    num_instances = data.shape[0]
    num_candidates = data.shape[1]
    num_rows = (num_score_types + num_cols - 1) // num_cols  # Calculate the number of rows needed

    # Extract the final score (the last score type) for all instances and candidates, and reshape the array
    final_scores = data[:, :, -1].reshape(num_instances * num_candidates)  # Reshape to (instances * candidates,)

    # Create figure with a grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 4 * num_rows))

    # Flatten axes to iterate through them easily
    axes = axes.flatten()

    for i in range(num_score_types):
        # Collect the scores for the i-th score type across all instances and candidates and reshape
        score_type_data = data[:, :, i].reshape(num_instances * num_candidates)  # Reshape scores to (instances * candidates,)
        
        # Scatter plot: x = score for this score type, y = final score (last score type)
        axes[i].scatter(score_type_data, final_scores, alpha=0.1, s=1, color='blue')
        axes[i].set_title(f'Model {i + 1} (zero-mean: {zero_mean})' if not layers else f'Layer {i + 1} (zero-mean: {zero_mean})')
        axes[i].set_xlabel(f'Score Model {i + 1}' if not layers else f'Score Layer {i + 1}')
        axes[i].set_ylabel(f'Score Model {data.shape[2]}' if not layers else f'Score Layer {data.shape[2]}' )

    # Remove empty subplots if num_score_types is not a perfect multiple of num_cols
    for j in range(num_score_types, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout to prevent label overlap
    plt.tight_layout()

    # Save the figure instead of showing it
    plt.savefig(plot_name)



if __name__ == "__main__":
    LAYERS = False
    ZERO_MEAN = True
    reshaped_train_matrix, mean_vector, cov_matrix, final_test_matrix = get_train_test_data(PATH, layers=LAYERS, zero_mean=ZERO_MEAN)

    # Plot distribution for a specific instance (e.g., instance 0)
    plot_score_distribution(final_test_matrix, num_cols=6, plot_name = f"normalized_score_distributions_layers_{LAYERS}_zero_mean_{ZERO_MEAN}")
    plot_scatter_score_vs_final(final_test_matrix, num_cols=6, plot_name = f"scatter_score_vs_final_grid_layers_{LAYERS}_zero_mean_{ZERO_MEAN}", zero_mean=ZERO_MEAN, layers=LAYERS)


# scatter plot jede score gegen 24. score



