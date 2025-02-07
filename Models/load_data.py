import os
import pandas as pd
import numpy as np
from pathlib import Path
import random

def load_all_data(path, features_list, features_sequence = ['SSXcore', 'IPLA', 'DAO_EDG7', 'RNT', 'DAI_EDG7', 'ECE_PF'], test_size = 0.15, mini_test_size = 0.7):
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
    random.seed(42)
    random.shuffle(features_list) # Shuffle the list of files to avoid bias
    test_size = int(len(features_list) * test_size)
    mini_test_size = int(len(features_list) * mini_test_size)
    test_list = features_list[:test_size]
    train_list = features_list[test_size:mini_test_size]
    mini_test_list = features_list[mini_test_size:]

    # Define directories using pathlib
    features_dir = Path(path) / "features"
    targets_dir  = Path(path) / "targets"
    
    # Lists to accumulate data from each file
    all_x_train = []
    all_x_test = []
    all_y_train = []
    all_y_test = []
    all_x_mini_test = []
    all_y_mini_test = []

    for feature_id in train_list:
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
        all_x_train.append(x)
        all_y_train.append(y)

    for feature_id in test_list:
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
        all_x_test.append(x)
        all_y_test.append(y)

    for feature_id in mini_test_list:
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
        all_x_mini_test.append(x)
        all_y_mini_test.append(y)

    all_x_train = np.vstack(all_x_train)
    all_y_train = np.concatenate(all_y_train)
    all_x_test = np.vstack(all_x_test)
    all_y_test = np.concatenate(all_y_test)
    all_x_mini_test = np.vstack(all_x_mini_test)
    all_y_mini_test = np.concatenate(all_y_mini_test)

    return all_x_train, all_y_train, all_x_test, all_y_test, all_x_mini_test, all_y_mini_test