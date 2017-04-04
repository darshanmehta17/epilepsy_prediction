import numpy as np
import pandas as pd 

def floor_half(x):
    '''
    Returns the greatest half lesser than the number.
    '''
    return 0.5 * np.floor(2.0 * x)

def index_of(arr, k):
    '''
    Returns the index of the value in the array.
    '''
    return np.nonzero(arr == k)[0][0]

def read_summary_file(filename):
    """
    Reads the summary file from location
    """
    df = pd.read_csv(filename,parse_dates=[1,2])
    return df