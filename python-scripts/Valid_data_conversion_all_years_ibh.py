import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from datetime import datetime
import time
import math


year_list = np.arange(2014, 2018).tolist()
for year in year_list:
    #Importing Data
    filename = "/project/lindner/moving/summer2018/Pollutant_Data/{}_data_ibh.csv".format(year)
    original_data = pd.read_csv(filename, low_memory=False)
    try:
        original_data = original_data.drop('Unnamed: 0', axis = 1)
    except:
        pass

    original_data['time'] = pd.to_datetime(original_data['epoch'], unit='s').dt.time
    original_data['date'] = pd.to_datetime(original_data['epoch'], unit='s').dt.date
    original_data['day'] = pd.to_datetime(original_data['epoch'], unit='s').dt.day
    original_data['month'] = pd.to_datetime(original_data['epoch'], unit='s').dt.month
    original_data['year'] = pd.to_datetime(original_data['epoch'], unit='s').dt.year
    original_data['hour']  = pd.to_datetime(original_data['epoch'], unit='s').dt.hour

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
    imp_cols = ['o3', 'temp', 'windspd', 'winddir']
    data = pd.DataFrame()
    data['epoch'] = epoch_list
    data['time'] = pd.to_datetime(data['epoch'], unit='s').dt.time
    data['day'] = pd.to_datetime(data['epoch'], unit='s').dt.day
    data['month'] = pd.to_datetime(data['epoch'], unit='s').dt.month
    data['year'] = pd.to_datetime(data['epoch'], unit='s').dt.year
    data['hour'] = pd.to_datetime(data['epoch'], unit='s').dt.hour
    data.set_index('epoch', inplace=True)
    
    for site in site_list:
        dummy_frame = sample_data.loc[sample_data['siteID']==site]
        dummy_frame.set_index('epoch', inplace=True)
        dummy_frame = dummy_frame[imp_cols]
        dummy_frame['prev_O3'] = np.nan
        dummy_frame['wind_x_dir'] = dummy_frame['windspd'] * np.cos(dummy_frame['winddir']*(np.pi/180))
        dummy_frame['wind_y_dir'] = dummy_frame['windspd'] * np.sin(dummy_frame['winddir']*(np.pi/180))
        dummy_frame.drop('windspd', axis=1, inplace = True)
        dummy_frame.drop('winddir', axis=1, inplace = True)
        dummy_frame = dummy_frame.add_prefix(site+'_')
        data = pd.merge(data, dummy_frame, how='left', left_index=True, right_index=True)
    
    data.reset_index()
    data_hour = data.groupby(['year','month','day', 'hour'], as_index=False).mean()
    data_hour = data.groupby(['year','month','day', 'hour'], as_index=False).mean()
    for site in site_list:
        data_prev = data_hour[site+'_o3'].tolist()
        data_prev.insert(0, np.nan)
        del data_prev[-1]
        data_hour[site+'_prev_O3'] = data_prev
    data_hour.isna().sum()

    data_hour.to_csv('{}_ozone_hourly.csv'.format(year))
    

    