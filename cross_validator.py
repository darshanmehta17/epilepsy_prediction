import random

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

from metrics import *
from naive_bayes_preditor import naive_bayes
from train import random_forest, svm_light
from utils import *


def load_data(summary, path, logger):
    logger.info('Loading data...')
    summary = summary[summary['Include'] == 1]

    files = [x.strip(' ') for x in summary['File Name']]

    logger.info('Total number of files = ' + str(len(files)))

    X = np.load(path + files[0] + '_data.npy')
    y = np.load(path + files[0] + '_target.npy')

    for file in files[1:]:
        X = np.append(X, np.load(path + file + '_data.npy'), axis=0)
        y = np.append(y, np.load(path + file + '_target.npy'), axis=0)

    # convert target values to -1 | 1
    y[y == 0] = -1

    # convert infs to 0
    X[X == np.inf] = 0
    X[X == -np.inf] = 0

    logger.info('Data loaded successfully')

    logger.info('Shuffling data...')
    X, y = unison_shuffled_copies(X, y)

    return X, y


def k_cross_validator(X, y, k, logger):
    logger.info('Splitting array...')
    indices = np.array(np.array_split(np.arange(X.shape[0]), k))

    logger.info('Starting cross validation...')
    for itr in range(k):
        logger.info('Validating set ' + str(itr))

        # Creating the splits
        X_test = X[indices[itr]]
        X_train = np.delete(X, indices[itr], axis=0)
        y_test = y[indices[itr]]
        y_train = np.delete(y, indices[itr], axis=0)

        logger.info('Running Naive Bayes...')
        naive_bayes(X_train, y_train, X_test, y_test, logger)

        logger.info('Running SVM...')
        svm_light(X_train, y_train, X_test, y_test, logger)

        logger.info('Running Random Forest...')
        random_forest(X_train, y_train, X_test, y_test, logger)

    logger.info('Cross validation succesfully completed.')


def main():
    logger = setup_logging('logs/', 'cross_validation')
    summary = read_summary_file('./input/patient_summary.csv')
    X, y = load_data(summary, 'D:/Tanay_Project/processed/', logger)

    k_cross_validator(X, y, 10, logger)


if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))
