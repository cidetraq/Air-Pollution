
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np


# In[11]:


def transform(year):
    #filename='Data_'+str(year)+'.csv'
    #
    #source_path='/project/lindner/moving/summer2018/Data_structure_3/'
    #data=pd.read_csv(source_path+filename)
    data=pd.read_csv('data08_100000.csv')
    data=data[data['nox_flag']=='VAL']
    data=data[data['no_flag']=='VAL']
    data=data[data['o3_flag']=="VAL"]
    data=data[data['temp_flag']=="VAL"]
    data=data[~data['winddir'].isna()]
    #Dropping unnecessary columns
    data=data.drop(['co_flag', 'humid', 'humid_flag', 'pm25', 'pm25_flag', 'so2', 'so2_flag', 'solar',  'solar_flag', 'dew', 'dew_flag', 'redraw', 'co', 'no_flag', 'no2_flag', 'nox_flag', 'o3_flag', 'winddir_flag', 'windspd_flag', 'temp_flag'], axis=1)
    data['hour']  = pd.to_datetime(data['epoch'], unit='s').dt.hour
    data['day'] = pd.to_datetime(data['epoch'], unit='s').dt.day
    data['month'] = pd.to_datetime(data['epoch'], unit='s').dt.month
    data['year'] = year
    data['wind_x_dir'] = data['windspd'] * np.cos(data['winddir']*(np.pi/180))
    data['wind_y_dir'] = data['windspd'] * np.sin(data['winddir']*(np.pi/180))
    #data.to_csv('Data_'+str(year)+'_ready.csv', index=False)
    data.to_csv('Data_'+str(year)+'_100k.csv', index=False)


# In[14]:


#years=np.arange(2000,2014)
years=[2008]
for year in years:
    transform(year)

