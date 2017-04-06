import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.externals import joblib
import os

from utils import *

summary = read_summary_file('input/patient_summary.csv')

test_files = ['chb01_26.edf','chb01_27.edf','chb01_29.edf']

path = 'D:/Tanay_Project/processed/'

files = [x.strip(' ') for x in summary['File Name'] if 'chb01' in x]

X_train = np.load(path+files[0]+'_data.npy')
y_train = np.load(path+files[0]+'_target.npy')
X_test = np.load(path+test_files[0]+'_data.npy')
y_test = np.load(path+test_files[0]+'_target.npy')
for file in test_files[:-1]:
    X_test = np.append(X_test,np.load(path+file+'_data.npy'),axis=0)
    y_test = np.append(y_test,np.load(path+file+'_target.npy'),axis=0)

for file in files[1:]:
    if file not in test_files:
        X_train = np.append(X_train, np.load(path+file+'_data.npy'),axis=0)
        y_train = np.append(y_train, np.load(path+file+'_target.npy'),axis=0)

print('Loaded Data')
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

model = SVC()
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)
model.fit(X_train,y_train)
joblib.dump(model,'svm.pkl')
# model = joblib.load('svm.pkl')

print('Training Done')

y_p = model.predict(X_test)
np.save(arr=y_p,file='output/y_predict')
np.save(arr=y_test,file='output/y_test')
print('accuracy_score = {:.3f}'.format(accuracy_score(y_test,y_p)*100))
print('f1_score = {:.3f}'.format(f1_score(y_test,y_p)*100))