import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

from metrics import *
from utils import *


def naive_bayes(X_train, y_train, X_test, y_test, logger):
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    [accuracy, recall, precision, f1_score] = evaluate_model(y_test, y_pred)
    dump_data_to_csv(
        np.array([accuracy, recall, precision, f1_score]), 'perf_naive_bayes.csv')
