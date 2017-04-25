import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

from metrics import *
from utils import *


def load_data(summary, path, logger):

	summary = summary[summary['Include'] == 1]
	summary_train = summary[summary['Test'] == 0]
	summary_test = summary[summary['Test'] == 1]

	train_files = [x.strip(' ') for x in summary_train['File Name']]
	test_files = [x.strip(' ') for x in summary_test['File Name']]

	logger.info('Len train files = '+str(len(train_files)))
	logger.info('Len test files = '+str(len(test_files)))

	X_train = np.load(path+train_files[0]+'_data.npy')
	y_train = np.load(path+train_files[0]+'_target.npy')
	X_test = np.load(path+test_files[0]+'_data.npy')
	y_test = np.load(path+test_files[0]+'_target.npy')

	for file in train_files[1:]:
		X_train = np.append(X_train, np.load(path+file+'_data.npy'),axis=0)
		y_train = np.append(y_train, np.load(path+file+'_target.npy'),axis=0)
	
	for file in test_files[1:]:
		X_test = np.append(X_test,np.load(path+file+'_data.npy'),axis=0)
		y_test = np.append(y_test,np.load(path+file+'_target.npy'),axis=0)

	logger.info('Loaded Data')
	logger.info('Train data shape = '+str(X_train.shape) + str(y_train.shape))
	logger.info('Test data shape = '+str(X_test.shape) + str(y_test.shape))

	return X_train, y_train, X_test, y_test

logger = setup_logging('logs/','naive_bayes')
summary = read_summary_file('./input/patient_summary.csv')
X_train, y_train, X_test, y_test = load_data(summary, './processed/', logger)

# convert target values to -1 | 1
y_train[y_train == 0] = -1
y_test[y_test==0] = -1

# convert infs to 0
X_train[X_train == np.inf] = 0
X_test[X_test == np.inf] = 0

X_train[X_train == -np.inf] = 0
X_test[X_test == -np.inf] = 0

print X_train.shape, y_train.shape, X_test.shape, y_test.shape

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d" \
       % (X_test.shape[0],(y_test != y_pred).sum()))

f1_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)