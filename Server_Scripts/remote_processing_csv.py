import os
import numpy as np
import pandas as pd
import random
from scipy.io import loadmat
import h5py
import pickle
import tempfile
import shutil

def convert_to_standard_format(data):
    """
    Recursively convert nested arrays with dtype=object to standard NumPy arrays or lists,
    and flatten them.
    """
    if isinstance(data, np.ndarray) and data.dtype == object:
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
        keys_jet = ['IP', 'WMHD', 'RNT', 'DAO_EDG7', 'SSXcore']
        for key in keys_jet:
            if key in mat_contents['SIG'].dtype.names:
                signal_data = mat_contents['SIG'][key][0][0]
                if 'time' in signal_data.dtype.names:
                    time_data = signal_data['time']
                else:
                    time_data = mat_contents['SIG']['time'][0][0]
                processed_data[key] = {
                    'signal': convert_to_standard_format(signal_data['signal']).flatten(),
                    'time': convert_to_standard_format(time_data).flatten()
                }

        # Check for additional keys in the .mat file
        if 'objDIS' in mat_contents:
            if 'disr_ipla_td' in mat_contents['objDIS'].dtype.names:
                processed_data['disr_ipla_td'] = mat_contents['objDIS']['disr_ipla_td'][0][0][0][1]
            else:
                print("Key 'disr_ipla_td' not found in mat_contents['objDIS']")

        if 'Discharge' in mat_contents:
            additional_keys = ['Ramp_up', 'Flat_top', 'Ramp_down']
            for key in additional_keys:
                try:
                    if key in mat_contents['Discharge'].dtype.names:
                        processed_data[key] = convert_to_standard_format(mat_contents['Discharge'][key][0][0][0])
                    else:
                        print("Key {} not found in mat_contents['Discharge']".format(key))
                except (IndexError, KeyError, AttributeError) as e:
                    print("Could not process key {}: {}".format(key, e))

        return processed_data

    except NotImplementedError:
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
        keys_jet = ['IP', 'WMHD', 'RNT', 'DAO_EDG7', 'SSXcore']
        for key in keys_jet:
            if key in h5_file['SIG']:
                signal_data = h5_file['SIG'][key]
                if 'time' in signal_data:
                    time_data = signal_data['time'][:]
                else:
                    time_data = h5_file['SIG']['time'][:]
                processed_data[key] = {
                    'signal': convert_to_standard_format(signal_data['signal'][:]).flatten(),
                    'time': convert_to_standard_format(time_data).flatten()
                }

        if 'objDIS' in h5_file:
            if 'disr_ipla_td' in h5_file['objDIS']:
                processed_data['disr_ipla_td'] = h5_file['objDIS']['disr_ipla_td'][:][1][0]
            additional_keys = ['Ramp_up', 'Flat_top', 'Ramp_down']
            for key in additional_keys:
                if key in h5_file['objDIS']['Discharge']:
                    processed_data[key] = h5_file['objDIS']['Discharge'][key][:].flatten()
                else:
                    print("Key {} not found in h5_file['objDIS']['Discharge']".format(key))

    return processed_data

def downsample_timeseries(begin_time, end_time, time_series, signal_series, length=1000):
    """
    Downsample a timeseries using interpolation within a specified time range.
    """
    # Filter time and signal within the specified range
    mask = (time_series >= begin_time) & (time_series <= end_time)
    time_series = time_series[mask]
    signal_series = signal_series[mask]

    # Handle case where no data points are in range
    if len(time_series) == 0:
        print("Warning: No data points in the specified time range.")
        return np.linspace(begin_time, end_time, length), np.zeros(length)

    # Create the downsampled time points
    downsampled_time = np.linspace(begin_time, end_time, length)

    # Interpolate the signal at the downsampled time points
    downsampled_signal = np.interp(downsampled_time, time_series, signal_series)

    return downsampled_time, downsampled_signal

def downsample_and_merge(shot, length=1000, keys=['SSXcore', 'IP', 'DAO_EDG7', 'WMHD', 'RNT']):
    """
    Downsample and merge time-series data for a single shot.
    """
    t_b, t_e = shot['Ramp_up'][0], shot['Ramp_down'][1]
    merged_df = pd.DataFrame()

    for key in keys:
        if key not in shot:
            continue

        if len(shot[key]['time']) == 0 or len(shot[key]['signal']) == 0:
            downsampled_time = np.linspace(t_b, t_e, length)
            downsampled_signal = np.zeros(length)
        else:
            downsampled_time, downsampled_signal = downsample_timeseries(
                t_b, t_e, shot[key]['time'], shot[key]['signal'], length
            )

        downsampled_df = pd.DataFrame({'time': downsampled_time, key: downsampled_signal})
        if merged_df.empty:
            merged_df = downsampled_df
        else:
            merged_df = pd.merge(merged_df, downsampled_df, on='time', how='inner')

    return merged_df

def process_and_save_as_csv(file_path, output_folder, length=1000):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(output_folder, "{}.csv".format(file_name))

    if os.path.exists(output_file):
        print("File {} already exists. Skipping...".format(output_file))
        return

    if file_path.endswith('.mat'):
        shot_data = process_mat_file(file_path)
    elif file_path.endswith('.h5'):
        shot_data = process_h5_file(file_path)
    else:
        print("Unsupported file format: {}".format(file_path))
        return

    try:
        merged_df = downsample_and_merge(shot_data, length=length)
        merged_df.to_csv(output_file, index=False)
        print("Saved downsampled data of file {} to {}".format(file_path, output_file))
    except Exception as e:
        print("Failed to process {}: {}".format(file_path, e))

def main():
    remote_path = "/Lac8_D/DEFUSE/DEFUSE_DB/DB_mat/"
    output_folder = "/home/tost/NoTivoli/downsampled_csvs/"

    # Check if the output folder exists; if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    remote_files = os.listdir(remote_path)
    jet_files = [file for file in remote_files if 'JET' in file and (file.endswith('.mat') or file.endswith('.h5'))]
    # random.shuffle(jet_files)  # Shuffle the list to get a statistical value
    # jet_files = jet_files[:10]  # Limit the number of files to process for testing

    for file_name in jet_files:
        file_path = os.path.join(remote_path, file_name)
        process_and_save_as_csv(file_path, output_folder)

if __name__ == "__main__":
    main()
