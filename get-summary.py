import pandas as pd
import os
import re
import numpy as np
from datetime import datetime, timedelta
import sys

def find_col(x):
    """
    Find column name from summary 
    """
    return x.split(":",1)[0]

def strip_col(x):
    """
    Strip away column name from summary
    """
    return x.split(":",1)[-1]

def parse_dates(x):
    fmt = '%H:%M:%S'
    x = x.strip(' ')
    x = x.split(":")
    if int(x[0]) >= 24 and int(x[0]) < 34:
        h = '0' + str(int(x[0]) - 24)
        x[0] = h
        x = ":".join(x)
        date = datetime.strptime(x,fmt)
        date = date + timedelta(days=1)
        return date
    x = ":".join(x)
    return datetime.strptime(x,fmt)

def get_patient_summary(summary):
    """
    Convert a patients summary to DF
    """
    df = pd.DataFrame()
    summary = summary.split('\n\n')
    for i in range(2,len(summary)):
        # Validate that we have the right record
        if summary[i].split('\n')[0].find('File Name:') < 0:
            continue
        df = df.append(pd.Series(summary[i].split('\n')),ignore_index=True)
    return df

def get_duration(summary):
    # Parse start and end times
    summary['File End Time'] = summary['File End Time'].apply(lambda x: parse_dates(x))
    summary['File Start Time'] = summary['File Start Time'].apply(lambda x: parse_dates(x))
    # Get duration
    summary['Duration'] = summary['File End Time'] - summary['File Start Time']
    # Get duration in seconds
    summary['Duration'] = summary['Duration'].apply(lambda x: x.seconds)
    return summary

def get_all_patients_summary(dataset_dir, output_dir):
    """
    Convert all patients summary
    """
    patients = [x for x in os.listdir(dataset_dir) if os.path.isdir(dataset_dir+x) and os.path.getsize(dataset_dir+x)>1000]
    df = pd.DataFrame()
    for patient in patients:
        with open(dataset_dir+os.sep+patient+os.sep+patient+'-summary.txt','r') as f:
            summary = f.read()
        df = df.append(get_patient_summary(summary))
    df.columns = df.dropna().iloc[0,:].apply(find_col)
    df = df.fillna("").applymap(strip_col)
    df.replace('',np.nan)
    df = get_duration(df)
    df.to_csv(output_dir+'patient_summary.csv',index=None)

if __name__ == '__main__':
    if len(sys.argv) > 2:
        get_all_patients_summary(sys.argv[1],sys,argv[2])
    else:
        get_all_patients_summary('/Users/tanay/chbmit/pn6/chbmit/','input/')

        