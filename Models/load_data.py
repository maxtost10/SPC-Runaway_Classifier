import os
import pandas as pd
import numpy as np
from pathlib import Path

def load_all_data(path, features_list, features_sequence = ['SSXcore', 'IPLA', 'DAO_EDG7', 'RNT', 'DAI_EDG7', 'ECE_PF']):
    """
    Loads and stacks features and targets from a list of CSV files.
    
    Parameters:
    - path: Base directory containing subdirectories 'features' and 'targets'.
    - features_list: List of file names to process.
    - features_sequence: List of feature names to extract from each CSV.
    
    Returns:
    - all_x: A 2D NumPy array containing all feature datapoints.
    - all_y: A 1D NumPy array containing all target values.
    """
    # Define directories using pathlib
    features_dir = Path(path) / "features"
    targets_dir  = Path(path) / "targets"
    
    # Lists to accumulate data from each file
    all_x = []
    all_y = []
    
    for feature_id in features_list:
        # Build full file paths
        feature_file = features_dir / feature_id
        target_file = targets_dir / feature_id
        
        # Load features CSV
        df_features = pd.read_csv(feature_file)
        time_length = len(df_features['time'])
        
        # Create a list for feature columns; if a key is missing, fill with zeros.
        x_list = []
        for key in features_sequence:
            if key in df_features.columns:
                x_list.append(df_features[key].to_numpy())
            else:
                print(f"Key {key} not in {feature_id}. Filling it with zeros.")
                x_list.append(np.zeros(time_length))
                
        # Stack features so that each row is a datapoint and each column a feature.
        # np.column_stack ensures the resulting array has shape (n_samples, len(features_sequence))
        x = np.column_stack(x_list)
        
        # Load target CSV and extract the 'target' column
        y = pd.read_csv(target_file)['target'].to_numpy()
        
        # Append this file's data to the overall lists
        all_x.append(x)
        all_y.append(y)
    
    # Combine all files' data into single arrays.
    # np.vstack stacks arrays vertically (along the first axis)
    all_x = np.vstack(all_x)
    # For targets, np.concatenate will combine 1D arrays into one long 1D array
    all_y = np.concatenate(all_y)
    
    return all_x, all_y