import os
from scipy.io import loadmat
import h5py
import cPickle as pickle
import numpy as np
import tempfile
import shutil

def convert_to_standard_format(data):
    """
    Recursively convert nested arrays with dtype=object to standard NumPy arrays or lists,
    and flatten them.
    """
    if isinstance(data, np.ndarray) and data.dtype == np.object:
        return np.array([convert_to_standard_format(item) for item in data]).flatten()
    elif isinstance(data, (list, tuple)):
        return [convert_to_standard_format(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_to_standard_format(value) for key, value in data.items()}
    else:
        return data

def process_mat_file(file_path):
    """
    Processes a .mat file and extracts the necessary data.

    Parameters:
        file_path (str): The path to the .mat file.

    Returns:
        dict: The processed data.
    """
    try:
        mat_contents = loadmat(file_path)
        processed_data = {}

        # Extract the necessary data from the .mat file
        keys_jet = ['IP', 'WMHD', 'RNT', 'DAI_EDG7']
        for key in keys_jet:
            if key in mat_contents['SIG'].dtype.names:
                signal_data = mat_contents['SIG'][key][0][0]
                processed_data[key] = {
                    'signal': convert_to_standard_format(signal_data['signal']).flatten()
                }

        return processed_data

    except NotImplementedError:
        # Handle MATLAB v7.3 files using h5py
        return process_h5_file(file_path)

def process_h5_file(file_path):
    """
    Processes a .h5 file and extracts the necessary data.

    Parameters:
        file_path (str): The path to the .h5 file.

    Returns:
        dict: The processed data.
    """
    processed_data = {}

    with h5py.File(file_path, 'r') as h5_file:
        keys_jet = ['IP', 'WMHD', 'RNT', 'DAI_EDG7']
        for key in keys_jet:
            if key in h5_file['SIG']:
                signal_data = h5_file['SIG'][key]
                processed_data[key] = {
                    'signal': convert_to_standard_format(signal_data['signal'][:]).flatten()
                }

    return processed_data

def save_to_pickle(data, file_name):
    """
    Saves the processed data to a Pickle file using a temporary file.

    Parameters:
        data (dict): The processed data.
        file_name (str): The name of the Pickle file.
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    try:
        with open(temp_file.name, 'wb') as pickle_file:
            pickle.dump(data, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        shutil.move(temp_file.name, file_name)
        print("Data saved to {}".format(file_name))
    finally:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)

def load_existing_data(file_name):
    """
    Loads existing data from a Pickle file if it exists.

    Parameters:
        file_name (str): The name of the Pickle file.

    Returns:
        dict: The loaded data, or an empty dictionary if the file does not exist.
    """
    if os.path.exists(file_name):
        with open(file_name, 'rb') as pickle_file:
            return pickle.load(pickle_file)
    return {}

def main():
    remote_path = "/Lac8_D/DEFUSE/DEFUSE_DB/DB_mat/"
    pickle_file_name = "all_JET_data.pkl"
    
    # Load existing data if it exists
    all_data = load_existing_data(pickle_file_name)
    processed_shots = set(all_data.keys())

    # List all files in the remote directory
    remote_files = os.listdir(remote_path)

    # Filter files to include only JET .mat and .h5 files
    jet_files = [file for file in remote_files if 'JET' in file and (file.endswith('.mat') or file.endswith('.h5'))]

    # jet_files = jet_files[:20]  # Limit the number of files to process for testing

    for file_name in jet_files:
        shot_number = file_name.split('.')[0]  # Extract shot number from file name
        if shot_number in processed_shots:
            print("Skipping already processed file: {}".format(file_name))
            continue

        file_path = os.path.join(remote_path, file_name)
        print("Processing file: {}".format(file_path))

        if file_name.endswith('.mat'):
            processed_data = process_mat_file(file_path)
        elif file_name.endswith('.h5'):
            processed_data = process_h5_file(file_path)

        all_data[shot_number] = processed_data

        # Save the aggregated data to a single Pickle file after each file is processed
        save_to_pickle(all_data, pickle_file_name)

    print("Aggregated Pickle file created: {}".format(pickle_file_name))

if __name__ == "__main__":
    main()