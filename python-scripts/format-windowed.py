#!/usr/bin/env python
# coding: utf-8

# In[ ]:


profiles={'cluster': {'input_source': '/project/lindner/moving/summer2018/2019/data-formatted/mark/', 'out_path': '/project/lindner/moving/summer2018/2019/data-intermediate/'},
          'nicholas': {'input_source': 'D:/programming-no-gdrive/air-pollution/data-formatted/', 'out_path': 'D:/programming-no-gdrive/air-pollution/data-intermediate/'} , 
          'carroll': {} }


# In[ ]:


import pandas as pd
import argparse
import matplotlib
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
from format_time_cat import time_cat 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# In[ ]:


def main(user, years, window, cat):
    if '-' in years:
        year_arr=years.split('-')
        years=np.arange(int(year_arr[0]), int(year_arr[1]))
    else:
        years=[int(years)]
    source_path=profiles[user]['input_source']
    out_path=profiles[user]['out_path']
    for year in years:
        windowed(source_path,year,out_path,window,cat)


# In[ ]:


def windowed(source_path,year,out_path,window_stride,cat):
    # Average window_stride elements together to form a single row
    data=pd.read_csv(source_path+str(year)+'_mark.csv')
    if cat==True:
        df=time_cat(data)
    if user!='carroll':
        df = df[df['val']=='y']
    sample_hours = window_stride / 12.0
    print("Sample Hours: %f" % sample_hours)

    # Number of future samples to mean for prediction
    prediction_window = int(24 / sample_hours)
    print("Prediction Window: %d" % prediction_window)

    # Length of the windowed sequence
    sequence_length = int(7*24 / sample_hours)
    print("Sequence Length: %d" % sequence_length)

    columns=df.columns
    # Number of features we take from the data
    input_features = len(columns)
    num_features = input_features
    num_inputs = input_features

    # Number of things we are doing regression to predict
    num_outputs = 4

    # Unprocessed dataset
    nd = df[columns].values

    # Windowed dataset
    nd_window = np.zeros((int(nd.shape[0] / window_stride), num_inputs))

    row = 0
    while row < nd.shape[0] - window_stride:
        for i in range(0, input_features):
            if cat==True:
                if i>2 and i<11:
                    nd_window[int(row/window_stride)][i] = np.mean(nd[row:row+window_stride,i])
                else:
                    nd_window[int(row/window_stride)][i] = mode(nd[row:row+window_stride,i])[0][0]
                    nd_window[int(row/window_stride)][i] = np.mean(nd[row:row+window_stride,i])
            else:
                nd_window[int(row/window_stride)][i] = np.mean(nd[row:row+window_stride,i])
        row += window_stride

    scaler = MinMaxScaler()
    scaler.fit(nd_window)
    nd_window = scaler.transform(nd_window)


    # Create sequences
    data = []
    labels = []


    rows = deque(maxlen=sequence_length)

    for idx, r in enumerate(nd_window):

        rows.append([a for a in r])

        # We need the entire sequence filled to make a prediction about the future mean
        if len(rows) < sequence_length:
            continue

        # Since we are predicting the mean, make sure we do not go out of bounds in the future
        if idx+1 + prediction_window > nd_window.shape[0]:
            break

        data.append(rows.copy())

        # We are predicting the future mean values
        u_24_no = np.mean( nd_window[idx+1 : idx+1 + prediction_window, 4] )
        u_24_no2 = np.mean( nd_window[idx+1 : idx+1 + prediction_window, 5] )
        u_24_nox = np.mean( nd_window[idx+1 : idx+1 + prediction_window, 6] )
        u_24_o3 = np.mean( nd_window[idx+1 : idx+1 + prediction_window, 7] )

        labels.append([u_24_no, u_24_no2, u_24_nox, u_24_o3])

    data = np.array(data)
    labels = np.array(labels) 
    data.dump(out_path+str(year)+'_data_windowed.ndarray')
    labels.dump(out_path+str(year)+'_labels_windowed.ndarray')


# In[ ]:


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", '--user', type=str,
                        help="cluster, nicholas, carroll")
    parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2],
                        help="does nothing at the moment")
    parser.add_argument('-y', '--years', type=str, help='type year range. full would be 2000-2018. or type single year without the dash.')
    parser.add_argument('-w', '--window', type=int, help='window size. starting point is 12')
    parser.add_argument('-c', '--categorical', type=bool, help='Run using time categorical features')
    args = parser.parse_args()
    main(args.user, args.years, args.window, args.categorical)

