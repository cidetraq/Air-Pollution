
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from datetime import datetime
import time
import math


# <h3>2014-2018</h3>

# In[ ]:


year_list = np.arange(2014, 2018).tolist()
for year in year_list:
    #Importing Data
    filename = "/project/lindner/moving/summer2018/Pollutant_Data/{}_data_ibh.csv".format(year)
    original_data = pd.read_csv(filename, low_memory=False, nrows=10)
    #Below use when ready 
    #original_data = pd.read_csv(filename, low_memory=False)
    try:
        original_data = original_data.drop('Unnamed: 0', axis = 1)
    except:
        pass
    data=pd.DataFrame()
    data['time'] = pd.to_datetime(original_data['epoch'], unit='s').dt.time
    data['date'] = pd.to_datetime(original_data['epoch'], unit='s').dt.date
    data['day'] = pd.to_datetime(original_data['epoch'], unit='s').dt.day
    data['month'] = pd.to_datetime(original_data['epoch'], unit='s').dt.month
    data['year'] = pd.to_datetime(original_data['epoch'], unit='s').dt.year
    data['hour']  = pd.to_datetime(original_data['epoch'], unit='s').dt.hour
    print(original_data.columns)
    quit()
    #List of epoch times
    epoch_start = original_data['epoch'].unique().min()
    epoch_end = original_data['epoch'].unique().max()
    epoch_list = list(np.arange(epoch_start, epoch_end+300, 300))

    #import available sites data
    site_data = pd.read_csv("Valid_sites.csv")
    site_list = list(site_data['O3'].dropna())

    try:
        site_data = site_data.drop('Unnamed: 0', axis = 1)
    except:
        pass

    #conversion of data
    sample_data = original_data
    data['epoch'] = epoch_list
    data.set_index('epoch', inplace=True)
    data['site'] = original_data['siteID']
    data['o3']=original_data['o3']
    data['temp']=original_data['temp']
    data['wind_x_dir'] = original_data['windspd'] * np.cos(original_data['winddir']*(np.pi/180))
    data['wind_y_dir'] = original_data['windspd'] * np.sin(original_data['winddir']*(np.pi/180))
    data_hour = data.groupby(['year','month','day', 'hour'], as_index=False).mean()
    data_hour = data.groupby(['year','month','day', 'hour'], as_index=False).mean()
    data_hour.to_csv('/home/narandal/{}_ozone_hourly_ds3.csv'.format(year))

