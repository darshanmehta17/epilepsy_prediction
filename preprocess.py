import pandas as pd
import numpy as np
import os

from data_generator import *

def read_summary_file(filename):
	df = pd.read_csv(filename,parse_dates=[1,2])
	return df

def main():
	summary = read_summary_file('patient_summary.csv')
	path = '/Users/tanay/chbmit/pn6/chbmit/' # Path to dataset dir

	for i in range(len(summary)):
		file_name = summary.iloc[i]['File Name']
		file_name = file_name.strip(' ')
		print file_name
		patient_name = file_name.split('_')[0]
		patient_name = patient_name.strip(' ')
		duration = summary.iloc[i]['Duration']
		n_seizures = summary.iloc[i]['Number of Seizures in File']

		filepath = path+patient_name+'/'+file_name
		print filepath
		data = generateFileData(filepath)

		# Sanity check to make sure size of data is correct
		assert data.shape[0] == duration-6, "Data size mismatch"
		print data.shape
		data = data.reshape(duration-6,-1)

		target = np.zeros(duration-6)

		if n_seizures == 0:
			np.save(file_name+'_data',data)
			np.save(file_name+'_target',target)

		for j in range(n_seizures):
			start = summary.iloc[i]['Seizure '+str(j+1)+' Start Time']
			end = summary.iloc[i]['Seizure '+str(j+1)+' End Time']

			start = int(start.strip(' seconds'))
			end = int(end.strip(' seconds'))

			assert start <= end, "Seizure start time more than end time"

			target[start:end] = 1

		np.save(file_name+'_data',data)
		np.save(file_name+'_target',target)

if __name__ == '__main__':
	main()