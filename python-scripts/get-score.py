from sklearn.externals import joblib
from collections import deque
from sklearn.preprocessing import MinMaxScaler
import pickle
import pandas as pd
import numpy as np

nd_window=pickle.load(open('/project/lindner/moving/summer2018/2019/data-intermediate/windowed_2000.pkl', 'rb'))

#Not going to scale it here yet
'''scaler = MinMaxScaler()
scaler.fit(nd_window)
nd_window = scaler.transform(nd_window)

# Create sequences
data = []
labels = []

window_stride = 12

sample_hours = window_stride / 12.0
print("Sample Hours: %f" % sample_hours)

# Number of future samples to mean for prediction
prediction_window = int(24 / sample_hours)
print("Prediction Window: %d" % prediction_window)

# Length of the windowed sequence
sequence_length = int(7*24 / sample_hours)
print("Sequence Length: %d" % sequence_length)

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
data = data.reshape((data.shape[0], data.shape[1]*data.shape[2]))
labels = np.array(labels)

'''
path='/project/lindner/moving/summer2018/2019/data-intermediate/'
data=pickle.load(path+'windowed_2000.pkl_data.ndarray')
la=pickle.load(path+'windowed_2000.pkl_data.ndarray')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.33, random_state=42)

model=joblib.load(open('/project/lindner/moving/summer2018/2019/models/rf_full_windowed_1ktrees_2000.pkl', 'rb'))
print(model.score(X_test, y_test))