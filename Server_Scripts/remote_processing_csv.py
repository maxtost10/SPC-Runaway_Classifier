import os
import numpy as np
import pandas as pd
import random
from scipy.io import loadmat
import h5py
import pickle
import tempfile
import shutil
from scipy.ndimage import gaussian_filter1d

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
        keys_jet = ['IPLA', 'WMHD', 'RNT', 'DAO_EDG7', 'SSXcore', 'DAI_EDG7', 'ECE_PF']
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
        keys_jet = ['IPLA', 'WMHD', 'RNT', 'DAO_EDG7', 'SSXcore', 'DAI_EDG7', 'ECE_PF']
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

def downsample_timeseries(begin_time, end_time, time_series, signal_series, timestep_size=1e-3, sigma=2):
    """
    Downsample a timeseries using interpolation within a specified time range.
    
    Parameters:
    - begin_time: Start time for the downsampling.
    - end_time: End time for the downsampling.
    - time_series: Array of time points from the original data.
    - signal_series: Corresponding signal values.
    - timestep_size: The desired time step between samples.
    - sigma: The standard deviation for the Gaussian filter.
    
    Returns:
    - downsampled_time: New time array with fixed time step.
    - downsampled_signal: Interpolated signal values at the new time points.
    """
    # Ensure valid input
    if end_time <= begin_time:
        raise ValueError("end_time must be greater than begin_time")

    # Create the new time points with fixed timestep
    downsampled_time = np.arange(begin_time, end_time, timestep_size)

    # Filter time and signal within the specified range
    mask = (time_series >= begin_time) & (time_series <= end_time)
    filtered_time_series = time_series[mask]
    filtered_signal_series = signal_series[mask]

    # Handle case where no data points are in range
    if len(filtered_time_series) == 0:
        print("Warning: No data points in the specified time range.")
        return downsampled_time, np.zeros(len(downsampled_time))

    # Interpolate the signal at the downsampled time points
    downsampled_signal = np.interp(downsampled_time, filtered_time_series, filtered_signal_series)

   # Smooth using a moving average
    downsampled_signal = gaussian_filter1d(downsampled_signal, sigma=sigma)  # Apply Gaussian smoothing

    return downsampled_time, downsampled_signal

def downsample_and_merge(shot, file_name, keys=['SSXcore', 'IPLA', 'DAO_EDG7', 'WMHD', 'RNT', 'DAI_EDG7', 'ECE_PF'], timestep_size=1e-3):
    """
    Downsample and merge time-series data for a single shot.
    Ensures all signals share the same time grid.
    """
    if np.isnan(shot['disr_ipla_td']) or shot['disr_ipla_td'] < shot['Ramp_up'][1]:
        print('Shot {} can not be processed since the disruption time is not given or too small.'.format(file_name))
        return pd.DataFrame()

    t_b, t_e = shot['disr_ipla_td'] - 1, shot['disr_ipla_td'] + 5  # Start and end times with respect to disruption time
    merged_df = pd.DataFrame()

    # Generate a shared time grid
    shared_time_grid = np.arange(t_b, t_e, timestep_size)

    for key in keys:
        if key not in shot or len(shot[key]['time']) == 0 or len(shot[key]['signal']) == 0:
            print('Signal {} from shot {} could not be downsampled since it was not available'.format(key, file_name))
            continue

        # Downsample using a fixed time grid
        downsampled_time, downsampled_signal = downsample_timeseries(
            t_b, t_e, shot[key]['time'], shot[key]['signal'], timestep_size=timestep_size
        )

        # Create DataFrame with the shared time grid
        downsampled_df = pd.DataFrame({'time': downsampled_time, key: downsampled_signal})

        # Merge on 'time' column
        if merged_df.empty:
            merged_df = downsampled_df
        else:
            merged_df = pd.merge(merged_df, downsampled_df, on='time', how='inner')

    return merged_df

def process_and_save_as_csv(file_path, output_folder):
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
        merged_df = downsample_and_merge(shot_data, file_name)
        if merged_df.empty:
            raise ValueError("")
        merged_df.to_csv(output_file, index=False)
        print("Saved downsampled data of file {} to {}".format(file_path, output_file))
    except Exception as e:
        print("Failed to process {}: {}".format(file_path, e))

def main():
    remote_path = "/Lac8_D/DEFUSE/DEFUSE_DB/DB_mat/"
    output_folder = "/home/tost/NoTivoli/downsampled_csvs_nodtIP_conv"

    # Check if the output folder exists; if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    remote_files = os.listdir(remote_path)
    jet_files = [file for file in remote_files if 'JET' in file and (file.endswith('.mat') or file.endswith('.h5'))]
    # random.shuffle(jet_files)  # Shuffle the list to get a statistical value
    #jet_files = jet_files[:10]  # Limit the number of files to process for testing

    for file_name in jet_files:
        file_path = os.path.join(remote_path, file_name)
        process_and_save_as_csv(file_path, output_folder)

if __name__ == "__main__":
    main()
