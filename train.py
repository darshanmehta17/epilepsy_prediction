import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.externals import joblib
import os

from utils import *

def load_data(file_summary, test_files, path, logger):
	summary = read_summary_file(file_summary)
	files = [x.strip(' ') for x in summary['File Name'] if 'chb01' in x]

	X_train = np.load(path+files[0]+'_data.npy')
	y_train = np.load(path+files[0]+'_target.npy')
	X_test = np.load(path+test_files[0]+'_data.npy')
	y_test = np.load(path+test_files[0]+'_target.npy')
	for file in files[1:4]:
	    if file in test_files[1:]:
	        X_test = np.append(X_test,np.load(path+test_files[0]+'_data.npy'),axis=0)
	        y_test = np.append(y_test,np.load(path+test_files[0]+'_target.npy'),axis=0)
	    else:
	        X_train = np.append(X_train, np.load(path+file+'_data.npy'),axis=0)
	        y_train = np.append(y_train, np.load(path+file+'_target.npy'),axis=0)

	logger.info('Loaded Data')
	logger.info('Train data shape = '+str(X_train.shape) + str(y_train.shape))
	logger.info('Test data shape = '+str(X_test.shape) + str(y_test.shape))

	return X_train, y_train, X_test, y_test

def main(logger):
	
	file_summary = 'input/patient_summary.csv'
	test_files = ['chb01_26.edf','chb01_27.edf','chb01_29.edf']
	path = 'D:/Tanay_Project/processed/'

	X_train, y_train, X_test, y_test = load_data(file_summary,test_files,path,logger)

	model = SVC()
	model.fit(X_train,y_train)
	joblib.dump(model,'svm.pkl')
	# model = joblib.load('svm.pkl')

	logger.info('Training Done')

	y_p = model.predict(X_test)
	logger.info('accuracy_score = {:.3f}'.format(accuracy_score(y_test,y_p)*100))
	logger.info('f1_score = {:.3f}'.format(f1_score(y_test,y_p)*100))

if __name__ == '__main__':
	logger = setup_logging('logs/')
	main()

