#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
import numpy as np


# In[2]:


import os
#local
os.chdir('../')


# In[52]:


#Cluster
#source_path='/project/lindner/moving/summer2018/Data_structure_3/'
#local
source_path='D:/programming-no-gdrive/air-pollution/data/'
#Local test
#source_path="D:/programming-no-gdrive/DASH/Air Pollution/data-sample/100k/"
#Cluster
#out_path='/project/lindner/moving/summer2018/2019/data-formatted/mark/'
#local
out_path='D:/programming-no-gdrive/air-pollution/data-formatted/mark/'


# In[54]:


def transform(source_path,year):
    filename='Data_'+str(year)+'.csv'
    #Local test
    #filename='data00_100000.csv'
    data=pd.read_csv(source_path+filename)
    data['hour']  = pd.to_datetime(data['epoch'], unit='s').dt.hour
    data['day'] = pd.to_datetime(data['epoch'], unit='s').dt.day
    data['month'] = pd.to_datetime(data['epoch'], unit='s').dt.month
    data['year'] = str(year)
    data['wind_x_dir'] = data['windspd'] * np.cos(data['winddir']*(np.pi/180))
    data['wind_y_dir'] = data['windspd'] * np.sin(data['winddir']*(np.pi/180))
    orig=data
    if year<2014:
        print('Year: '+str(year)+ ' '+'Initial length of data: ')
        print(len(data))
        data=data[data['nox_flag']=='VAL']
        data=data[data['no_flag']=='VAL']
        data=data[data['o3_flag']=="VAL"]
        print('Year: '+str(year)+ ' '+'Length after o3 restriction: ')
        print(len(data))
        data=data[data['temp_flag']=="VAL"]
        print('Year: '+str(year)+ ' '+"Length after temp flag restriction: ")
        print(len(data))
        data=data[~data['winddir'].isna()]
        print('Year: '+str(year)+ ' '+'Length after winddir nulls gone: ')
        print(len(data)) 
    if year>=2014:
        print('Year: '+str(year)+ ' '+'Initial length of data: ')
        print(len(data))
        data=data[data['nox_flag']=='VAL']
        data=data[data['no_flag']=='VAL']
        data=data[data['o3_flag']=="K"]
        print('Year: '+str(year)+ ' '+'Length after o3 restriction: ')
        print(len(data))
        data=data[data['temp_flag']=="K"]
        print('Year: '+str(year)+ ' '+"Length after temp flag restriction: ")
        print (len(data))
        data=data[~data['winddir'].isna()]
        print('Year: '+str(year)+ ' '+'Length after winddir nulls gone: ')
        print(len(data)) 
    data=index(orig,data)
    data=data.drop(['co_flag', 'humid', 'humid_flag', 'pm25', 'pm25_flag', 'so2', 'so2_flag', 'solar',  'solar_flag', 'dew', 'dew_flag', 'redraw', 'co', 'no_flag', 'no2_flag', 'nox_flag', 'o3_flag', 'winddir_flag', 'windspd_flag', 'temp_flag'], axis=1)
    data.to_csv(out_path+str(year)+'_mark.csv', index=False)


# In[55]:


def index(orig, data):
    good_indices=data.index
    orig['val']=np.nan 
    good=orig.loc[good_indices].replace({'val': np.nan}, 'y')
    bad_indices=orig.index.difference(good_indices)
    bad=orig.loc[bad_indices].replace({'val': np.nan}, 'n')
    orig=orig.drop(['val'], axis=1)
    val=pd.concat([good, bad], axis=0, join='inner')
    val=val.sort_index()
    return val


# In[56]:


#years=np.arange(2000,2018)
years=[2000]
for year in years:
    transform(source_path,year)

