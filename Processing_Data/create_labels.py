import os
import pandas as pd
import numpy as np

def load_and_process_data(base_path, re_autom_path, re_valid_path, check_nans_infs):
    """
    Loads CSV files into a dictionary, filters RE shot lists, and extracts feature names.

    Parameters:
    - base_path (str): Path to the directory containing the CSV files.
    - re_autom_path (str): Path to the automatic RE numbers CSV file.
    - re_valid_path (str): Path to the validated RE numbers CSV file.
    - check_nans_infs (function): Function to check and drop NaNs/Infs.

    Returns:
    - dataframes (dict): Dictionary with shot numbers as keys and dataframes as values.
    - RE_autom (list): Filtered list of automatic RE shot numbers.
    - RE_valid (list): Filtered list of validated RE shot numbers.
    - NO_RE_probably (list): List of shots that are neither in RE_autom nor RE_valid.
    - features (list): List of feature names (excluding 'time').
    """

    # Load all CSV files into a dictionary with shot numbers as keys
    dataframes = {
        int(file.split('.')[0].split('no')[1]): pd.read_csv(os.path.join(base_path, file))
        for file in os.listdir(base_path)
    }

    # Load RE shot lists
    RE_autom = list(pd.read_csv(re_autom_path, header=None)[0])
    RE_valid = list(pd.read_csv(re_valid_path, header=None)[0])

    # Check and clean NaNs/Infs
    check_nans_infs(dataframes)

    # Update shot lists to include only those present in dataframes
    RE_autom = [shot for shot in RE_autom if shot in dataframes]
    RE_valid = [shot for shot in RE_valid if shot in dataframes]
    NO_RE_probably = [shot for shot in dataframes if shot not in RE_autom and shot not in RE_valid]

    # Extract feature names (excluding 'time')
    features = list(dataframes[NO_RE_probably[0]].keys()) if NO_RE_probably else []
    if 'time' in features:
        features.remove('time')

    return dataframes, RE_autom, RE_valid, NO_RE_probably, features


def save_re_targets(RE_lifetimes, base_path_re, save_path_targets):
    '''
    Creates targets for training in a network later on. All timesteps that are within the runaway time window will be set to 1, those outside to 0.

    Parameters:
    - RE_lifetimes (dict): Dictionary with shot numbers of runaway positive shots and their lifetime window(s).
                           If each shot has multiple lifetime intervals, it should be a list of tuples [(start1, end1), (start2, end2), ...].
    - base_path_re (str): Path to the folder containing the CSV files with the shot parameters, including those in RE_lifetimes.
    - save_path_targets (str): Path where the targets should be saved as csv.

    Returns:
    - targets (dict): Dictionary where:
        - Keys: Shot numbers (same as in RE_lifetimes).
        - Values: NumPy arrays (binary masks, same length as the time series of each shot).
                  Each element is 1 if the timestep is within any runaway phase, otherwise 0.
    '''

    # Ensure that the save path exists
    os.makedirs(save_path_targets, exist_ok=True)

    targets = {}

    for shot_nr, lifetimes in RE_lifetimes.items():
        # Load CSV file
        file_path = os.path.join(base_path_re, f'JETno{shot_nr}.csv')
        data = pd.read_csv(file_path)
        time = data['time']  # Convert to NumPy array for efficiency
        
        target = np.zeros(len(time))

        # Store target in dictionary
        target[(time>lifetimes[0]) & (time<lifetimes[1])]=1
        targets[shot_nr] = np.copy(target)

        # Save to CSV immediately
        df = pd.DataFrame({'time': time, 'target': np.copy(target)})
        df.to_csv(os.path.join(save_path_targets, f"JETno{shot_nr}.csv"), index=False)

    return targets

def save_no_re_targets(NO_RE_numbers, base_path, save_path_targets):
    '''
    Creates targets for training in a network later on. All timesteps in the target will be set to 0, because we assume that no re are present in the data.

    Parameters:
    - NO_RE_numbers (list or array-like): Array with shot numbers of runaway negative shots.
    - base_path (str): Path to the folder containing the CSV files with the shot parameters.
    - save_path_targets (str): Path where the targets should be saved as csv.

    Returns:
    - targets (dict): Dictionary where:
        - Keys: Shot numbers (same as in NO_RE_numbers).
        - Values: Array with zeros, the same length as the corresponding time array of the shot
    '''

    # Ensure save path exists
    os.makedirs(save_path_targets, exist_ok=True)

    targets = {}

    for shot_nr in NO_RE_numbers:
        # Load CSV file
        file_path = os.path.join(base_path, f'JETno{shot_nr}.csv')
        data = pd.read_csv(file_path)
        time = data['time'].values  # Convert to NumPy array for efficiency

        target = np.zeros(len(time))

        # Store target in dictionary
        targets[shot_nr] = target

        # Save to CSV
        df = pd.DataFrame({'time': time, 'target': target})
        df.to_csv(os.path.join(save_path_targets, f"JETno{shot_nr}.csv"), index=False)

    return targets