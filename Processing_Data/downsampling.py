def downsample_timeseries(begin_time, end_time, time_series, signal_series, length, method='mean'):
    # Create a DataFrame from the time and signal series
    df = pd.DataFrame({'time': time_series, 'signal': signal_series})
    
    # Filter the DataFrame for the given time range
    df = df[(df['time'] >= begin_time) & (df['time'] <= end_time)]
    
    # Calculate the box size
    box_size = len(df) // length
    
    # Downsample by averaging or median over the box size
    downsampled_time = []
    downsampled_signal = []
    
    for i in range(0, len(df), box_size):
        box = df.iloc[i:i + box_size]
        downsampled_time.append(box['time'].iloc[0])
        if method == 'mean':
            downsampled_signal.append(box['signal'].mean())
        elif method == 'median':
            downsampled_signal.append(box['signal'].median())
    
    return downsampled_time, downsampled_signal