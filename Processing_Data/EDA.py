import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np
from scipy.integrate import simps  # For numerical integration
import matplotlib.pyplot as plt

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
                if len(dataframes[RE_key][feature]) == 0:
                    print(f"DataFrame '{RE_key}' is empty for feature '{feature}'.")
                    continue
                # Extract the feature values as a NumPy array and append to temp
                values = dataframes[RE_key][feature].to_numpy()
                if values.size > 0:  # Ensure the array is not empty
                    temp.append(values)
            except Exception as e:
                print(f"Error processing feature '{feature}' in DataFrame '{RE_key}': {e}")
                continue  # Skip this DataFrame and proceed to the next one

        # Ensure temp is not empty before computing min and max
        if temp:
            # Flatten temp into a single array
            temp_flat = np.concatenate([arr for arr in temp if arr.size > 0])
            features_extrema[feature] = (np.min(temp_flat), np.max(temp_flat))
        else:
            print(f"Warning: No valid data for feature '{feature}'")
            features_extrema[feature] = (None, None)  # Handle case where no data is available

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
import numpy as np

def check_nans_infs(dataframes):
    """
    Checks each DataFrame in the given dictionary for NaN (Not a Number)
    and Inf (Infinity) values. Replaces NaN values with zeros.
    
    Parameters:
    - dataframes (dict): A dictionary where the keys are identifiers for
      DataFrames and the values are pandas DataFrame objects.
      
    Returns:
    - None: The function prints the count of NaNs and Infs for each DataFrame.
    """
    
    # Iterate over each DataFrame in the dictionary
    for key, df in dataframes.items():
        # Count the total number of NaNs in the DataFrame
        nans = df.isna().sum().sum()
        
        # Count the total number of Infs in the DataFrame
        infs = np.isinf(df).sum().sum()
        
        # If the DataFrame contains NaNs, replace them with zeros and print a message
        if nans > 0:
            print(f"DataFrame {key}: {nans} NaNs found. Replacing NaNs with 0.")
            df.fillna(0, inplace=True)
        
        # If the DataFrame contains Infs, print the count
        if infs > 0:
            print(f"DataFrame {key}: {infs} Infs found.")

def check_nans_infs_pdf(dataframes, drop=True):
    """
    Checks each DataFrame in the given dictionary for NaN (Not a Number)
    and Inf (Infinity) values. Replaces NaN values with zeros.
    
    Parameters:
    - dataframes (dict): A dictionary where the keys are identifiers for
      DataFrames and the values are pandas DataFrame objects.
      
    Returns:
    - None: The function prints the count of NaNs and Infs for each DataFrame.
    """
    
    # Iterate over each DataFrame in the dictionary
    for key, df in dataframes.items():
        # Count the total number of NaNs in the DataFrame
        nans = df.isna().sum().sum()
        
        # Count the total number of Infs in the DataFrame
        infs = np.isinf(df).sum().sum()
        
        # If the DataFrame contains NaNs, replace them with zeros and print a message
        if nans > 0:
            if drop:
                print(f"DataFrame {key}: {nans} NaNs found. Dropping rows with NaNs.")
                df.dropna(inplace=True)
        
        # If the DataFrame contains Infs, print the count
        if infs > 0:
            if drop:
                print(f"DataFrame {key}: {infs} Infs found. Dropping them.")
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                df.dropna(inplace=True)




def plot_jet_data(RE_DICT: pd.DataFrame, NO_RE_DICT: pd.DataFrame, save_path: str, x_lim_re=None, x_lim_no_re=None):
    """
    Plots plasma parameters for RE and NO-RE cases from given DataFrames.
    
    Parameters:
    RE_DICT (pd.DataFrame): DataFrame containing runaway electron data with time column.
    NO_RE_DICT (pd.DataFrame): DataFrame containing non-runaway electron data with time column.
    save_path (str): File path to save the plot as SVG.
    x_lim_re (tuple): Limits for x-axis for RE plots (min, max).
    x_lim_no_re (tuple): Limits for x-axis for NO-RE plots (min, max).
    """
    keys_jet = ['IPLA', 'WMHD', 'RNT', 'DAI_EDG7', 'SSXcore'] #, 'DAO_EDG7'
    units = ['A', 'J', 'Counts', 'p/s/cm²/sr', 'p/s/cm²/sr', 'W/m$^2$']
    fs = 8
    
    fig, axes = plt.subplots(len(keys_jet), 2, figsize=(8, 8))
    
    for n, key in enumerate(keys_jet):
        axes[n, 0].plot(RE_DICT['time'], RE_DICT[key], label='RE')
        axes[n, 1].plot(NO_RE_DICT['time'], NO_RE_DICT[key], label='NO-RE')
        
        # Set x-limits
        if x_lim_re:
            axes[n, 0].set_xlim(x_lim_re)
        if x_lim_no_re:
            axes[n, 1].set_xlim(x_lim_no_re)

        # Set tick label font size
        axes[n, 0].tick_params(axis="both", which="major", labelsize=fs)
        axes[n, 1].tick_params(axis="both", which="major", labelsize=fs)
        
        # Set y-label with correct font size
        axes[n, 0].set_ylabel(f"{key[:3]} [{units[n]}]", fontsize=fs)
    
    # Set x-label with correct font size
    for ax in axes.flat:
        ax.set_xlabel("time [s]", fontsize=fs)
    
    plt.subplots_adjust(hspace=0.7)  # Increase vertical spacing
    
    # Save as Vector Graphic (SVG)
    plt.savefig(save_path, format="svg")