import os
import pandas as pd

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
    check_nans_infs(dataframes, drop=True)

    # Update shot lists to include only those present in dataframes
    RE_autom = [shot for shot in RE_autom if shot in dataframes]
    RE_valid = [shot for shot in RE_valid if shot in dataframes]
    NO_RE_probably = [shot for shot in dataframes if shot not in RE_autom and shot not in RE_valid]

    # Extract feature names (excluding 'time')
    features = list(dataframes[NO_RE_probably[0]].keys()) if NO_RE_probably else []
    if 'time' in features:
        features.remove('time')

    return dataframes, RE_autom, RE_valid, NO_RE_probably, features
