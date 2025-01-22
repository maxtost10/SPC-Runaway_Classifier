import os
from scipy.io import loadmat
import h5py
import random

def check_keys_in_mat_file(file_path, missing_keys_count):
    """
    Checks for the presence of specific keys in a .mat file.

    Parameters:
        file_path (str): The path to the .mat file.
        missing_keys_count (dict): Dictionary to count missing keys.
    """
    try:
        mat_contents = loadmat(file_path)

        # Check for the presence of keys in the .mat file
        keys_jet = ['disr_ipla_td']
        for key in keys_jet:
            if key not in mat_contents['Ramp_up', 'Flat_top', 'Ramp_down'].dtype.names:
                missing_keys_count[key] += 1

    except NotImplementedError:
        # Handle MATLAB v7.3 files using h5py
        check_keys_in_h5_file(file_path, missing_keys_count)

def check_keys_in_h5_file(file_path, missing_keys_count):
    """
    Checks for the presence of specific keys in a .h5 file.

    Parameters:
        file_path (str): The path to the .h5 file.
        missing_keys_count (dict): Dictionary to count missing keys.
    """
    with h5py.File(file_path, 'r') as h5_file:
        keys_jet = ['disr_ipla_td']
        for key in keys_jet:
            if key not in h5_file['Ramp_up', 'Flat_top', 'Ramp_down']:
                missing_keys_count[key] += 1

def main():
    remote_path = "/Lac8_D/DEFUSE/DEFUSE_DB/DB_mat/"
    
    # List all files in the remote directory
    remote_files = os.listdir(remote_path)

    # Filter files to include only JET .mat and .h5 files
    jet_files = [file for file in remote_files if 'JET' in file and (file.endswith('.mat') or file.endswith('.h5'))][:200] # Only process 200 files because it takes too long otherwise
    random.shuffle(jet_files) # Shuffling the list to get a statistical value

    # Initialize missing keys count
    keys_jet = ['disr_ipla_td']
    missing_keys_count = {key: 0 for key in keys_jet}

    for file_name in jet_files:
        file_path = os.path.join(remote_path, file_name)
        print("Processing file: {}".format(file_path))

        if file_name.endswith('.mat'):
            check_keys_in_mat_file(file_path, missing_keys_count)
        elif file_name.endswith('.h5'):
            check_keys_in_h5_file(file_path, missing_keys_count)

    print("Missing keys count: {}".format(missing_keys_count))
    print("Total files processed: {}".format(len(jet_files)))

if __name__ == "__main__":
    main()