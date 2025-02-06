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
    def __init__(self, data_path, features_list, features_sequence=None, transform=None, seq_length=6000, chunk_size=1):
        """
        Loads each CSV file, splits sequences into smaller chunks, and reshapes them to (60, 600).

        Parameters:
        - data_path (str or Path): Base directory containing 'features' and 'targets' subdirectories.
        - features_list (list of str): List of CSV file names to process.
        - features_sequence (list of str): List of feature names to extract from each CSV.
        - transform (callable, optional): Optional transform to be applied on the feature data.
        - seq_length (int): Expected original sequence length before splitting.
        - chunk_size (int): The number of time steps per chunk (default=100).
        """

        random.shuffle(features_list)  # Shuffle dataset order

        if features_sequence is None:
            features_sequence = ['SSXcore', 'IPLA', 'DAO_EDG7', 'RNT', 'DAI_EDG7', 'ECE_PF']
        
        self.samples = []  # Stores (x, y) tuples
        self.transform = transform
        self.chunk_size = chunk_size  # How many time steps per chunk
        self.num_chunks = seq_length // chunk_size  # 6000 // 100 = 60 chunks per sequence

        features_dir = Path(data_path) / "features"
        targets_dir  = Path(data_path) / "targets"

        # Process each CSV file
        for feature_id in features_list:
            feature_file = features_dir / feature_id
            target_file  = targets_dir / feature_id
            
            # Load features
            df_features = pd.read_csv(feature_file)
            time_length = len(df_features['time'])

            # Drop unexpected sequence lengths
            if time_length != seq_length:
                print(f'Skipping {feature_id}: sequence length {time_length} is unexpected.')
                continue
            
            # Extract feature columns or use zeros if missing
            x_list = []
            for key in features_sequence:
                if key in df_features.columns:
                    x_list.append(df_features[key].to_numpy())
                else:
                    x_list.append(np.zeros(time_length))
            
            # Convert to (time_length, num_features) = (6000, 6)
            x_file = np.column_stack(x_list)

            # Load targets
            y_file = pd.read_csv(target_file)['target'].to_numpy()  # Shape: (time_length,)

            # Reshape into (60, 100, 6) and then flatten 2nd axis → (60, 600)
            x_reshaped = x_file.reshape(self.num_chunks, self.chunk_size, -1)  # (60, 100, 6)
            x_final = x_reshaped.reshape(self.num_chunks, -1)  # (60, 600)

            # Do the same with targets (no flattening)
            y_reshaped = y_file.reshape(self.num_chunks, self.chunk_size)  # (60, 100)

            # Store each new sequence as a sample
            self.samples.append((x_final, y_reshaped))

        print(f"Processed {len(self.samples)} sequences with shape {x_final.shape}.")

    def __len__(self):
        """Return the total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Return a single sample (x, y) with tensor conversion."""
        sample, target = self.samples[idx]
        
        if self.transform:
            sample = self.transform(sample)

        # Convert to PyTorch tensors
        sample = torch.tensor(sample, dtype=torch.float32)  # (60, 600)
        target = torch.tensor(target, dtype=torch.float32)  # (60, 100)

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
        # out = self.sigmoid(out)

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



class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, num_heads=8):
        """
        Transformer model for sequence classification.

        Parameters:
        - input_size (int): Number of input features per time step.
        - hidden_size (int): Hidden dimension of the Transformer model.
        - num_layers (int): Number of Transformer encoder layers.
        - output_size (int): Number of output units (1 for binary classification).
        - dropout (float): Dropout probability.
        - num_heads (int): Number of attention heads.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Linear layer to project input to Transformer hidden size
        self.input_projection = nn.Linear(input_size, hidden_size)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=num_heads, 
            dim_feedforward=hidden_size * 4,  # Feedforward dimension
            dropout=dropout, 
            batch_first=True  # Ensures input shape: (batch, seq_len, hidden_size)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the Transformer model.

        Parameters:
        - x (torch.Tensor): Input tensor of shape [batch_size, sequence_length, input_size]

        Returns:
        - out (torch.Tensor): Output tensor of shape [batch_size, sequence_length, output_size]
        """
        # Project input to hidden_size dimension
        x = self.input_projection(x)  # Shape: (batch, seq_len, hidden_size)

        # Pass through Transformer Encoder
        x = self.transformer_encoder(x)  # Shape: (batch, seq_len, hidden_size)

        # Fully connected layers
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        # Remove last dimension to match target size
        return x.squeeze(-1)  # Binary values per timestep

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
