import logging
import os

import numpy as np
import pandas as pd


def dump_data_to_csv(metrics, filename):
    '''
    Saves the numpy array metrics to the given filename in csv format
    '''
    df = pd.DataFrame(metrics.reshape(1, -1))
    header = ['accuracy', 'recall', 'precision', 'f1_score']
    if not os.path.isfile(filename):
        df.to_csv(filename, header=header)
    else:
        df.to_csv(filename, mode='a', header=False)


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
    df = pd.read_csv(filename, parse_dates=[1, 2])
    df = df[df['Include'] == 1]
    return df


def setup_logging(logdir, name, log_level=logging.INFO):
    """
    Sets up logging
    """
    from datetime import datetime
    from os import makedirs
    from os.path import isdir

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if not isdir(logdir):
        makedirs(logdir)

    # File handler
    fh = logging.FileHandler(logdir + name + '.log')
    fh.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def unison_shuffled_copies(a, b):
    assert len(a) == len(
        b), "Length of the arrays to be shuffled is not the same"
    p = np.random.permutation(len(a))
    return a[p], b[p]
