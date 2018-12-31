#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import argparse 


# In[2]:


profiles={'cluster': {'input_source': '/project/lindner/moving/summer2018/Data_structure_3/', 'out_path': '/project/lindner/moving/summer2018/2019/data-formatted/mark/'},
          'nicholas': {'input_source': 'D:/programming-no-gdrive/air-pollution/data/', 'out_path': 'D:/programming-no-gdrive/air-pollution/data-formatted/mark/'} , 
          'carroll': {} }


# In[3]:


def main(user, years):
    if profiles[user]=='nicholas':
        import os
        #local
        os.chdir('../')
    #years=np.arange(2000,2018)
    if '-' in years:
        year_arr=years.split('-')
        years=np.arange(int(year_arr[0]), int(year_arr[1]))
    else:
        years=[years]
    source_path=profiles[user]['input_source']
    out_path=profiles[user]['out_path']
    for year in years:
        transform(source_path,year,out_path)


# In[4]:


def transform(source_path,year,out_path):
    filename='Data_'+str(year)+'.csv'
    data=pd.read_csv(source_path+filename)
    data['hour']  = pd.to_datetime(data['epoch'], unit='s').dt.hour
    data['day'] = pd.to_datetime(data['epoch'], unit='s').dt.day
    data['month'] = pd.to_datetime(data['epoch'], unit='s').dt.month
    data['year'] = str(year)
    try:
        data=data[data['month']!='month']
        data=data[data['day']!='day']
        data=data[data['hour']!='hour']
    except BaseException:
        pass
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


# In[5]:


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


# In[9]:


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", '--user', type=str,
                        help="cluster, nicholas, carroll")
    parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2],
                        help="increase output verbosity")
    parser.add_argument('-y', '--years', type=str, help='type year range. full would be 2000-2018')
    args = parser.parse_args()
    main(args.user, args.years)


# In[8]:


get_ipython().run_line_magic('tb', '')

