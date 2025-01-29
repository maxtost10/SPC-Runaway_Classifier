import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np
from scipy.integrate import simps  # For numerical integration

def compute_feature_statistics(dataframes, RE_valid, features):
    """
    Computes the minimum and maximum values (extrema) for each feature across multiple DataFrames.
    Additionally, estimates the probability density function (PDF) using Kernel Density Estimation (KDE) 
    and normalizes each density function before averaging.

    Parameters:
    - dataframes (dict): A dictionary where keys are identifiers and values are pandas DataFrames.
    - RE_valid (list): A list of keys corresponding to DataFrames that should be considered for analysis.
    - features (list): A list of feature names (column names) to analyze.

    Returns:
    - features_extrema (dict): A dictionary containing the min and max values for each feature.
    - features_densities (dict): A dictionary where each feature maps to an array of x-values (range)
      and an averaged, normalized density function computed from the available data.

    Notes:
    - If an error occurs while processing a feature in a DataFrame, it is skipped, and an error message is printed.
    """

    # Dictionary to store the min and max values of each feature across all valid DataFrames
    features_extrema = {}

    # Compute the extrema for each feature
    for feature in features:
        temp = []  # Temporary list to store feature values from different DataFrames
        
        for RE_key in RE_valid:
            try:
                # Extract the feature values as a NumPy array and store them in temp
                temp.append(dataframes[RE_key][feature].to_numpy())
            except Exception as e:
                print(f"Error processing feature '{feature}' in DataFrame '{RE_key}': {e}")
                continue  # Skip this DataFrame and proceed to the next one
        
        # Store the global minimum and maximum values of the feature
        features_extrema[feature] = (np.min(temp), np.max(temp))

    # Dictionary to store the density estimates for each feature
    features_densities = {}

    # Compute the Kernel Density Estimation (KDE) for each feature
    for feature in features:
        temp = []  # Temporary list to store estimated densities

        # Retrieve the min and max values for the feature
        min_val, max_val = features_extrema[feature]

        for RE_key in RE_valid:
            try:
                # Compute the KDE for the feature values
                kde = gaussian_kde(dataframes[RE_key][feature].to_numpy())
                
                # Generate x-values over the feature's range for evaluation
                x = np.linspace(min_val, max_val, 1000)

                # Evaluate the KDE on the generated x-values
                y = kde(x)

                # Normalize the KDE values so that the integral sums to 1
                integral = simps(y, x)  # Compute numerical integration using Simpson’s rule
                if integral > 0:  # Avoid division by zero
                    y /= integral

                # Store the normalized density estimate
                temp.append(y)
            except Exception as e:
                print(f"Error processing feature '{feature}' in DataFrame '{RE_key}': {e}")
                continue  # Skip this DataFrame and proceed to the next one

        # Store the averaged, normalized density function over all DataFrames for the feature
        if temp:  # Ensure temp is not empty before averaging
            features_densities[feature] = [x, np.mean(np.array(temp), axis=0)]
        else:
            print(f"Warning: No valid density estimates for feature '{feature}'")
            features_densities[feature] = [x, np.zeros_like(x)]  # Default to zero if no data available

    return features_extrema, features_densities


# Function to check for NaNs and Infs in a dictionary of DataFrames
def check_nans_infs(dataframes, drop=False):
    """
    Checks for NaN (Not a Number) and Inf (Infinity) values in each DataFrame 
    within a given dictionary of DataFrames. Optionally, it can remove 
    DataFrames that contain NaNs or Infs.

    Parameters:
    - dataframes (dict): A dictionary where keys are DataFrame names 
      (or identifiers) and values are pandas DataFrame objects.
    - drop (bool, optional): If True, removes DataFrames containing NaNs 
      or Infs from the dictionary. Default is False.

    Returns:
    - None: The function prints the count of NaNs and Infs for each DataFrame. 
      If `drop=True`, it also removes affected DataFrames from the dictionary 
      and prints the names of the dropped DataFrames.
    """

    # List to store keys of DataFrames that should be dropped
    keys_to_drop = []

    # Iterate over the dictionary of DataFrames
    for key, df in dataframes.items():
        # Count the total number of NaNs in the DataFrame
        nans = df.isna().sum().sum()

        # Count the total number of Infs in the DataFrame
        infs = np.isinf(df).sum().sum()

        # If the DataFrame contains NaNs or Infs, print the counts
        if nans > 0 or infs > 0:
            print(f"DataFrame {key}: NaNs = {nans}, Infs = {infs}")
            
            # If drop is enabled, mark this DataFrame for removal
            if drop:
                keys_to_drop.append(key)

    # If drop is enabled, remove DataFrames that contain NaNs or Infs
    if drop:
        for key in keys_to_drop:
            del dataframes[key]  # Remove the DataFrame from the dictionary
        print(f"Dropped DataFrames: {keys_to_drop}")