import numpy as np


'''
    Returns the greatest half lesser than the number 
'''
def floor_half(x):
    return 0.5 * np.floor(2.0 * x)

'''
    Returns the index of the value in the array
'''
def index_of(arr, k):
    return np.nonzero(arr == k)[0][0]
