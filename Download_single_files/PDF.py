import numpy as np

import matplotlib.pyplot as plt


def clean_data(pd_series):
    """
    Cleans the input pandas Series by removing any NaN values.

    Args:
        pd_series: pandas Series or list-like object containing numerical data.

    Returns:
        cleaned_array: NumPy array of the input data with NaN values removed.

    Example:
        import pandas as pd
        pd_series = pd.Series([1.0, 2.0, np.nan, 3.0, np.nan, 4.0])
        cleaned_data = clean_data(pd_series)
        print(cleaned_data)
        # Output: array([1.0, 2.0, 3.0, 4.0])
    """
    series_array = np.array(pd_series)
    return series_array[~np.isnan(series_array)]


def PDF(y, plot=False, norm=False):
    """
    Returning the probability distribution of the y values of a function, normalized to the average value if norm=True.

    Args:
        y: array-like
        norm: Boolean

    Returns:
        pdf: tuple ([pdf values], [pdf x_axis]) - normalized histogram of y
        scaling: Average of the y values rescaling the pdf
        
    Example:
        y = [1, 2, 3]
        (pdf, x), scaling = PDF(y)
        >>> ((array([0.33333333, 0.33333333, 0.33333333]),array([1.33333333, 2.        , 2.66666667])),1)
    """
    
    # Convert data to array without nans
    if type(y[0])!=float:
        y = clean_data(y)
    
    # Compute the histogram for the given values
    hist, bin_edges = np.histogram(y, bins='auto')
    
    # Normalising histogram
    hist = hist/len(y)
    
    # Calculate the bin centers for the x-axis (midpoints of bin edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    scaling=1
    
    # If norm=True, scale the PDF by the average
    if norm:
        # Calculate the average of the y values
        scaling = np.mean(y)
        bin_centers = bin_centers / scaling
        
    if plot:
        plt.plot(bin_centers, hist, label="scaling = {scal}".format(scal = scaling))
        if scaling != 1:
            plt.legend()

    return (hist, bin_centers), scaling


import numpy as np

def Edward(y, t, t_c):
    """
    Cuts the function values y into corresponding pieces at given time points, approximating
    where necessary based on the closest transition points in the time series. Referring to Edward Scissorhands.

    Args:
        y: array length N with function values
        t: array length N with timestamps
        t_c: array length < N with points where y should be cut (can include values not in t)
        
    Returns:
        y_cutted: array of length len(t_c)+1 containing the cut y data
        t_cutted: array of length len(t_c)+1 containing the cut t data
        
    Example:
        y = [1, 1, 1, 2, 2, 2, 2, 3, 3, 3]
        t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        t_c = [3, 7.5]
        Edward(y, t, t_c)
        >>> [[1, 1, 1], [2, 2, 2, 2, 3], [3, 3]], [[1, 2, 3], [4, 5, 6, 7, 8], [9, 10]]
    """
    
    # Ensure inputs are numpy arrays
    t_array = t.to_numpy()
    y = y.to_numpy()
    t_c_array = np.array(t_c)

    # Initialize the lists for storing cut segments
    y_cutted = []
    t_cutted = []
    
    # Initial index to start slicing from
    start_idx = 0
    
    for cut_point in t_c_array:
        # Check if cut_point is outside the range of t
        if cut_point < t_array[0] or cut_point > t_array[-1]:
            raise ValueError(f"Cut point {cut_point} is out of bounds for t.")
        
        # Find the index of the closest value in t to the cut point
        closest_idx = (np.abs(t_array - cut_point)).argmin()
        
        # Append the segment up to the closest index
        y_cutted.append(y[start_idx:closest_idx + 1])
        t_cutted.append(t[start_idx:closest_idx + 1])
        
        # Update the start index
        start_idx = closest_idx + 1
    
    # Add the final segment after the last cut point
    y_cutted.append(y[start_idx:])
    t_cutted.append(t[start_idx:])
    
    return y_cutted, t_cutted















