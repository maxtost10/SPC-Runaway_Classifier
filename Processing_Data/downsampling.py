import numpy as np
import pandas as pd

def downsample_timeseries(begin_time, end_time, time_series, signal_series, length):
    # Create a DataFrame from the time and signal series
    df = pd.DataFrame({'time': time_series, 'signal': signal_series})
    
    # Filter the DataFrame for the given time range
    df = df[(df['time'] >= begin_time) & (df['time'] <= end_time)]
    
    # Generate a consistent time axis using linspace
    downsampled_time = np.linspace(begin_time, end_time, length)
    
    # Create bins for the time intervals
    bins = np.linspace(begin_time, end_time, length + 1)
    
    # Assign each time value to a bin
    df['bin'] = pd.cut(df['time'], bins, labels=False, include_lowest=True)
    
    # Group by the bins and calculate the mean for each bin
    downsampled_signal = df.groupby('bin')['signal'].mean().values
    
    return downsampled_time, downsampled_signal

def downsample_and_merge(t_b, t_e, shot, length, keys= ['SSXcore', 'IP', 'DAO_EDG7', 'WMHD', 'RNT']):
    # Initialize an empty DataFrame for the merged result
    merged_df = pd.DataFrame()
    
    for key in keys:
        # Check if the timeseries is empty
        if len(shot[key]['time']) == 0 or len(shot[key]['signal']) == 0:
            downsampled_time = np.linspace(t_b, t_e, length)
            downsampled_signal = np.zeros(length)
        else:
            # Downsample the timeseries for the current key
            downsampled_time, downsampled_signal = downsample_timeseries(t_b, t_e, shot[key]['time'], shot[key]['signal'], length)
        
        # Create a DataFrame for the downsampled data
        downsampled_df = pd.DataFrame({'time': downsampled_time, key: downsampled_signal})
        
        # Merge with the existing merged DataFrame
        if merged_df.empty:
            merged_df = downsampled_df
        else:
            merged_df = pd.merge(merged_df, downsampled_df, on='time', how='inner')
    
    return merged_df