import os
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F  # Import for one-hot encoding
import torch.nn as nn
import random

class IndependentCSVDataset(Dataset):
    def __init__(self, data_path, features_list, features_sequence=None, transform=None, seq_length=6000, num_classes=2):
        """
        Loads each CSV file and stores individual samples (row-wise) as tuples of (x, y), 
        while ensuring targets are one-hot encoded.

        Parameters:
        - data_path (str or Path): Base directory containing 'features' and 'targets' subdirectories.
        - features_list (list of str): List of CSV file names to process.
        - features_sequence (list of str): List of feature names to extract from each CSV.
          Defaults to ['SSXcore', 'IPLA', 'DAO_EDG7', 'RNT', 'DAI_EDG7', 'ECE_PF'] if not provided.
        - transform (callable, optional): Optional transform to be applied on the feature data.
        - seq_length (int): Ensures all sequences have the same length.
        - num_classes (int): Number of classes for one-hot encoding.
        """

        random.shuffle(features_list) # Shuffle the list of features to avoid overfitting on the order of the features

        if features_sequence is None:
            features_sequence = ['SSXcore', 'IPLA', 'DAO_EDG7', 'RNT', 'DAI_EDG7', 'ECE_PF']
        
        self.samples = []  # List to hold individual (x, y) tuples
        self.transform = transform
        self.num_classes = num_classes  # Store number of classes for one-hot encoding
        
        # Define directories using pathlib
        features_dir = Path(data_path) / "features"
        targets_dir  = Path(data_path) / "targets"
        
        # Process each CSV file in the provided list
        for feature_id in features_list:
            feature_file = features_dir / feature_id
            target_file  = targets_dir / feature_id
            
            # Load the features CSV
            df_features = pd.read_csv(feature_file)
            time_length = len(df_features['time'])

            # Drop sequences with a length different from the desired one
            if time_length != seq_length:
                print(f'Skipping {feature_id}: sequence length {time_length} is unexpected.')
                continue
            
            # Build the feature matrix for the file.
            # For each key in features_sequence, use the column if available, otherwise use zeros.
            x_list = []
            for key in features_sequence:
                if key in df_features.columns:
                    x_list.append(df_features[key].to_numpy())
                else:
                    x_list.append(np.zeros(time_length))
            
            # x_file is a 2D array with shape (time_length, number_of_features)
            x_file = np.column_stack(x_list)
            
            # Load the targets CSV and extract the 'target' column
            y_file = pd.read_csv(target_file)['target'].to_numpy()  # shape: (time_length,)
            
            # Append each shot to the samples list.
            self.samples.append((x_file, y_file))
    
    def __len__(self):
        """Return the total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Return a single sample as a tuple (x, y) with one-hot encoded target."""
        sample, target = self.samples[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        # Convert the sample to PyTorch tensor
        sample = torch.tensor(sample, dtype=torch.float32)

        # Convert target to tensor and apply one-hot encoding
        target = torch.tensor(target, dtype=torch.float32)  # Ensure integer values
        
        return sample, target



def compute_global_minmax(dataset):
    """
    Computes global min and max for each feature across all time series in the dataset.
    
    Assumes that each sample from the dataset is a tuple (sample, target) where
    sample is a torch.Tensor of shape [sequence_length, num_features].
    
    Returns:
    - feature_min: NumPy array of shape (num_features,)
    - feature_max: NumPy array of shape (num_features,)
    """
    all_samples = []
    for sample, _ in dataset:
        if isinstance(sample, torch.Tensor):
            sample = sample.numpy()
        all_samples.append(sample)
    
    all_data = np.concatenate(all_samples, axis=0)
    
    # Compute min and max along axis 0 (for each feature)
    feature_min = np.min(all_data, axis=0)
    feature_max = np.max(all_data, axis=0)
    
    # Avoid division by zero by ensuring feature_max is strictly greater than feature_min
    zero_variance_mask = (feature_max == feature_min)
    feature_max[zero_variance_mask] = feature_min[zero_variance_mask] + 1e-6
    
    return feature_min, feature_max


class GlobalMinMaxNormalize:
    def __init__(self, min_vals, max_vals):
        """
        min_vals: array-like of shape (num_features,)
        max_vals: array-like of shape (num_features,)
        """
        # Convert min and max values to torch tensors for compatibility.
        self.min_vals = torch.tensor(min_vals, dtype=torch.float32)
        self.max_vals = torch.tensor(max_vals, dtype=torch.float32)
    
    def __call__(self, sample):
        """
        Normalizes the input sample using the global min and max values.
        sample: torch.Tensor of shape [sequence_length, num_features]
        Returns the normalized sample.
        """
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample, dtype=torch.float32)
        return (sample - self.min_vals) / (self.max_vals - self.min_vals)



import torch

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward propagate LSTM
        out, (_, _) = self.lstm(x)

        # Fully connected layers
        out = self.fc1(out)
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        # Remove last dimension to match target size
        return out.squeeze(-1)  # Binary values per timestep

    def predict(self, x, threshold=0.5, device="cpu"):
        """
        Predict binary outputs using a threshold.

        Parameters:
        - x (torch.Tensor): Input tensor of shape [batch_size, sequence_length, input_size]
        - threshold (float): Decision threshold (default=0.5)
        - device (str): Device ('cuda' or 'cpu')

        Returns:
        - Binary predictions (0 or 1) as a torch.Tensor of shape [batch_size, sequence_length]
        - Raw probabilities (before thresholding)
        """
        self.eval()  # Set model to evaluation mode
        x = x.to(device)

        with torch.no_grad():  # No gradients needed for inference
            probs = self.forward(x)  # Get probabilities from forward pass

        # Apply threshold to convert probabilities into binary predictions
        predictions = (probs > threshold).int()

        return predictions, probs  # Return both predictions and raw probabilities



def compute_class_weights(train_loader, num_classes=2):
    """
    Iterates through the train_loader, counts the occurrences of each class, 
    and computes class weights for imbalanced classification.

    Args:
        train_loader (DataLoader): PyTorch DataLoader with training data.
        num_classes (int): Number of unique classes. Default is 2 (binary classification).

    Returns:
        torch.Tensor: Ratio of class 0 to class 1 (to be used as `pos_weight` in BCEWithLogitsLoss).
    """
    class_counts = torch.zeros(num_classes)  # Initialize count array

    # Accumulate counts for class 0 and class 1
    for _, targets in train_loader:
        class_counts[1] += targets.sum().item()  # Count 1s
        class_counts[0] += (targets.numel() - targets.sum().item())  # Count 0s

    print(f"Class counts: {class_counts.tolist()}")
    print(f"Class ratio: {class_counts[0] / class_counts[1]}")

    # Convert to tensor (important for compatibility with BCEWithLogitsLoss)
    return torch.tensor(class_counts[0] / class_counts[1], dtype=torch.float32)
