import os
import h5py
import pickle
import numpy as np

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
                    'signal': convert_to_standard_format(signal_data['signal'][:]).flatten(),
                    'time': convert_to_standard_format(signal_data['time'][:]).flatten()
                }

    return processed_data

def save_to_pickle(data, file_name):
    """
    Saves the processed data to a Pickle file.

    Parameters:
        data (dict): The processed data.
        file_name (str): The name of the Pickle file.
    """
    with open(file_name, 'wb') as pickle_file:
        pickle.dump(data, pickle_file)
    print("Data saved to {}".format(file_name))

def main():
    remote_path = "/Lac8_D/DEFUSE/DEFUSE_DB/DB_mat/"
    
    # List all files in the remote directory
    remote_files = os.listdir(remote_path)

    # Filter files to include only JET .h5 files
    jet_h5_files = [file for file in remote_files if 'JET' in file and file.endswith('.h5')]

    # Process only the first 10 JET .h5 files
    jet_h5_files = jet_h5_files[:10]

    all_data = {}

    for file_name in jet_h5_files:
        file_path = os.path.join(remote_path, file_name)
        print("Processing file: {}".format(file_path))
        shot_number = file_name.split('.')[0]  # Extract shot number from file name

        processed_data = process_h5_file(file_path)
        all_data[shot_number] = processed_data

    # Save the aggregated data to a single Pickle file
    pickle_file_name = "all_JET_h5_data_first_10.pkl"
    save_to_pickle(all_data, pickle_file_name)
    print("Aggregated Pickle file created: {}".format(pickle_file_name))

if __name__ == "__main__":
    main()