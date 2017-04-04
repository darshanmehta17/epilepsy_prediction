import pandas as pd
import numpy as np
import os

from data_generator import *
from utils import *

def main():
	summary = read_summary_file('input/patient_summary.csv')
	path = 'D:/Tanay_Project/chbmit/' # Path to dataset dir
	output_dir = 'D:/Tanay_Project/processed/'
	window_size = 3
	epoch_size = 2

	for i in range(len(summary)):
		file_name = summary.iloc[i]['File Name']
		file_name = file_name.strip(' ')
		print(file_name)
		patient_name = file_name.split('_')[0]
		patient_name = patient_name.strip(' ')
		duration = summary.iloc[i]['Duration']
		n_seizures = summary.iloc[i]['Number of Seizures in File']

		filepath = os.path.join(path,patient_name,file_name)
		print(filepath)
		data = generateFileData(filepath)

		# Sanity check to make sure size of data is correct
		assert data.shape[0] == duration-6, "Data size mismatch"
		print(data.shape)
		data = data.reshape(duration-6,-1)

		target = np.zeros(duration-6)

		data_path = os.path.join(output_dir,file_name+'_data')
		target_path = os.path.join(output_dir,file_name+'_target')

		if n_seizures == 0:
			np.save(data_pathN,data)
			np.save(target_path,target)

		for j in range(n_seizures):
			start = summary.iloc[i]['Seizure '+str(j+1)+' Start Time']
			end = summary.iloc[i]['Seizure '+str(j+1)+' End Time']

			start = int(start.strip(' seconds'))-1
			end = int(end.strip(' seconds'))-1

			assert start <= end, "Seizure start time more than end time"

			target[start:end] = 1

		np.save(data_path,data)
		np.save(target_path,target)

if __name__ == '__main__':
	main()