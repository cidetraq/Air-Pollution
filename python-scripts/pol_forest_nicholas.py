#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Average window_stride elements together to form a single row
window_stride = 12

sample_hours = window_stride / 12.0
print("Sample Hours: %f" % sample_hours)

# Number of future samples to mean for prediction
prediction_window = int(24 / sample_hours)
print("Prediction Window: %d" % prediction_window)

# Length of the windowed sequence
sequence_length = int(7*24 / sample_hours)
print("Sequence Length: %d" % sequence_length)


# In[ ]:


print('Using already created windowed predictions for year 2000')


# # Read the data
# df = pd.read_csv('D:/programming-no-gdrive/air-pollution/data-formatted/ready/Data_2000_ready.csv')

# #formatting if needed
# df['month']=df['month'].apply(lambda x: str(x))
# df['hour']=df['hour'].apply(lambda x: str(x))
# df['day']=df['day'].apply(lambda x: str(x))
# pre_dummies=pd.concat([df['hour'], df['day'], df['month']], axis=1)
# dummies=pd.get_dummies(pre_dummies)
# df=pd.concat([df, dummies], axis=1)
# df=df.drop_duplicates()
# df=df.drop(['hour', 'day', 'month', 'temp_flag'], axis=1)

# # Drop bad rows
# #df = df[df['val']=='y']
# from scipy.stats import mode
# # Input Features
# columns=df.columns
# 
# # Number of features we take from the data
# input_features = len(columns)
# num_features = input_features
# num_inputs = input_features
# 
# # Number of things we are doing regression to predict
# num_outputs = 4
# 
# # Unprocessed dataset
# nd = df[columns].values
# 
# # Windowed dataset
# nd_window = np.zeros((int(nd.shape[0] / window_stride), num_inputs))
# 
# row = 0
# while row < nd.shape[0] - window_stride:
#     for i in range(0, input_features):
#         if i>2 and i<11:
#             nd_window[int(row/window_stride)][i] = np.mean(nd[row:row+window_stride,i])
#         else:
#             nd_window[int(row/window_stride)][i] = mode(nd[row:row+window_stride,i])[0][0]
#     row += window_stride
#     
# nd_window 

# In[ ]:


import pickle
nd_window=pickle.load(open('/project/lindner/moving/summer2018/2019/data-intermediate/windowed_2000.pkl', 'rb'))


# In[26]:


#Not going to scale it here yet
'''scaler = MinMaxScaler()
scaler.fit(nd_window)
nd_window = scaler.transform(nd_window)
'''
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
data = data.reshape((data.shape[0], data.shape[1]*data.shape[2]))
labels = np.array(labels)

print(data.shape)
print(labels.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.33, random_state=42)


# In[ ]:


n_trees=1000


# In[29]:


from sklearn.ensemble import RandomForestRegressor

regr = RandomForestRegressor(random_state=0, n_estimators=n_trees, n_jobs=-1, verbose=1)
regr.fit(X_train, y_train)
print(str(n_trees)+' tree regressor R^2 score: ')
regr.score(X_test, y_test) 


# In[31]:


from sklearn.externals import joblib
# Save to file in the current working directory
#Local
#joblib_file = "rf-no-all-2000_2_sample.pkl"  
joblib_file = "rf_full_windowed_1ktrees_2000.pkl" 
#Local
#joblib.dump(regr, 'D:/programming-no-gdrive/air-pollution/models/'+joblib_file)
joblib.dump(regr, '/project/lindner/moving/summer2018/2019/models/'+joblib_file)
#score=regr.score(X_test, y_test) 
#Local
#pickle.dump(score, open('D:/programming-no-gdrive/air-pollution/models/rf_full_windows_2000_score.pkl', 'wb'))
#score_file=open('/project/lindner/moving/summer2018/2019/descriptive-output/rf_full_windowed_1k_2000_score.txt', 'w')
#open('D:/programming-no-gdrive/air-pollution/models/rf_full_windows_2000_score.txt', 'w').write(str(score))


# #Plotting code below
# 
# plt.rcParams['figure.figsize'] = (20, 10)
# plt.rcParams['font.size'] = 16
# 
# for seq in range(0, data.shape[0] - sequence_length):
#     
#     lookup = {'no': (0, 0), 'no2':(0, 1), 'nox':(1, 0), 'o3':(1, 1)}
# 
#     pred = regr.predict(data[seq].reshape((1, 1512)))[0]
#     fig, ax = plt.subplots(2, 2)
# 
#     for idx,f in enumerate([(4, 'no'), (5, 'no2'), (6, 'nox'), (7, 'o3')]):
#     
#         feature_index, feature_name = f
#         
#         X = []
#         Y_actual = []
# 
#         for i in range(0, sequence_length + int(24*(1/sample_hours))):
#             X.append(seq+i)
#             Y_actual.append(data[seq+i][feature_index])
# 
#         Y_actual = np.array(Y_actual)
#         
#         predicted_mean = pred[feature_index - 4]
#         actual_mean = np.mean(Y_actual[sequence_length:])
#         rolling_mean = np.mean(Y_actual[:sequence_length])
#         rolling_std = np.std(Y_actual[:sequence_length])
#                 
#         Y_pred = Y_actual.copy()
#         Y_pred[sequence_length:] = predicted_mean
#         Y_pred[:sequence_length] = np.nan
# 
#         Y_actual_mean = Y_actual.copy()
#         Y_actual_mean[sequence_length:] = actual_mean
#         Y_actual_mean[:sequence_length] = np.nan
#         
#         Y_rolling_mean = Y_actual.copy()
#         Y_rolling_mean[:sequence_length] = rolling_mean
#         Y_rolling_mean[sequence_length:] = np.nan
#         
#         Y_rolling_std_upper = Y_actual.copy()
#         Y_rolling_std_upper[:sequence_length] = rolling_mean + rolling_std
#         Y_rolling_std_upper[sequence_length:] = np.nan
#         
#         Y_rolling_std_lower = Y_actual.copy()
#         Y_rolling_std_lower[:sequence_length] = rolling_mean - rolling_std
#         Y_rolling_std_lower[sequence_length:] = np.nan   
#         
#         subplot = ax[lookup[feature_name][0]][lookup[feature_name][1]]
# 
#         subplot.plot(X, Y_actual, color='black', linewidth=4.0)
#         subplot.plot(X, Y_actual_mean, color='green', linewidth=4.0)
#         subplot.plot(X, Y_pred, color='purple', linewidth=4.0)
#         subplot.plot(X, Y_rolling_mean, color='green', linewidth=4.0)
#         subplot.plot(X, Y_rolling_std_upper, color='orange', linewidth=4.0)
#         subplot.plot(X, Y_rolling_std_lower, color='orange', linewidth=4.0)
#         
#         subplot.grid()
#         
#         subplot.set_title("%s 24 hour mean prediction" % (feature_name,))
#         
#         subplot.set_xlabel("Hours")
#         subplot.set_ylabel("Scaled Concentration")
#     
#     fig.legend(['Actual Continuous', 'Actual Mean', 'Predicted Mean', 'Rolling Mean', 'Standard Deviation'])
#     fig.tight_layout()
# 
#     plt.savefig('charts/%.05d.png' % seq)
#     # plt.show()
#     plt.close()
# 
#     print("Rendered %d" % seq)

# In[29]:


name = {}

f = 0
for s in range(0, sequence_length):
    key = s - (sequence_length-1)
    name[f+0] = "%d_hour" % key
    name[f+1] = "%d_temp" % key
    name[f+2] = "%d_windspd" % key
    name[f+3] = "%d_winddir" % key
    name[f+4] = "%d_no" % key
    name[f+5] = "%d_no2" % key
    name[f+6] = "%d_nox" % key
    name[f+7] = "%d_o3" % key
    
    f += 8
    
    if fft_features:
        for i in range(0, 8):
            for r in range(0, 12):
                name[f] = "%d_fft_%d_%d" % (key, i, r)
                f += 1

pairs = []

for idx, imp in enumerate(regr.feature_importances_):    
    pairs.append([imp, name[idx]])
    
    
pairs.sort(reverse=True)
for v in pairs:
    value, key = v
    print("%s:\t\t\t%f" % (key, value))


# In[ ]:




