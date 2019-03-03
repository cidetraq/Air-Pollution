#!/usr/bin/env python
# coding: utf-8

# <h1>Objective</h1>
# See, for each column in data we care about, which sites have null values across entire columns; this will tell us if perhaps there are sites with sensors that are completely down for certain measurements at certain times or across the entire year (2000).

# In[ ]:


import pandas as pd
import numpy as np


# <h2>Make sure to change paths before uploading to cluster

# In[ ]:


descr_path = "/project/lindner/moving/summer2018/2019/descriptive-output/exploration/"
#on cluster
data = pd.read_csv('/project/lindner/moving/summer2018/Data_structure_3/Data_2000.csv')
#on local
#data = pd.read_csv('D:/programming-no-gdrive/air-pollution/data-sample/100k/data00_100000.csv')


# <h2>See data columns</h2>

# In[ ]:


#data.columns


# <h2>Output all AQS Codes (suspect some bad data)

# with open(descr_path+'codes.txt', 'w') as file:
#     for aqs in data['AQS_Code'].unique():
#         file.write(str(aqs)+"\n")

# <h2>Drop unneeded columns</h2>

# In[ ]:


data=data.drop(['redraw', 'co', 'co_flag', 'humid', 'humid_flag', 'pm25', 'pm25_flag', 'so2', 'so2_flag', 'solar', 'solar_flag', 'temp', 'temp_flag', 'dew', 'dew_flag', 'winddir_flag', 'windspd_flag', 'no_flag', 'no2_flag', 'nox_flag', 'o3_flag'], axis=1)


# <h2>Add time features (Hour, Day, Month). Code from format-mark in notebooks/format

# In[ ]:


data['hour']  = pd.to_datetime(data['epoch'], unit='s').dt.hour
data['day'] = pd.to_datetime(data['epoch'], unit='s').dt.day
data['month'] = pd.to_datetime(data['epoch'], unit='s').dt.month


# <h2>Drop all sites other than the 71 "good" sites Chinmay found with at least one data point in a relevant column per month

# In[ ]:


#On cluster
valid_sites = pd.read_csv("/project/lindner/moving/summer2018/2019/descriptive-output/Sites_with_monthly_data.csv")
#On local
#valid_sites = pd.read_csv("D:/programming-no-gdrive/DASH/Air Pollution/descriptive-output/Sites_with_monthly_data.csv")
data = data.loc[data['AQS_Code'].isin(valid_sites['site'])]


# In[ ]:


#Just doing hours first
#This does hours as rows
print("Working with Chinmay's 71 \"valid\" sites with at least one data point per month on selected columns \n")
all_site = pd.DataFrame()
for aqs_code in list(data['AQS_Code'].unique()):
    site_records = data[data["AQS_Code"] == aqs_code]
    print("Data for site: "+str(aqs_code)+ '\n')
    this_aqs_null_tallies = pd.DataFrame()
    for col in ['no', 'no2', 'nox', 'o3']:
        #pol_tallies = pd.DataFrame()
        hour_tallies = {}
        print("Pollutant: "+col+"\n")
        for month in range(1,13):
            #fill out
            for day in range(1,32):
                #fill out 
                for hour in range(0,24):
                    if hour not in hour_tallies: 
                        hour_tallies[hour] = 0
                    else:
                        this_hour = site_records[(site_records['day']==day) & (site_records['hour']==hour) & (site_records['month']==month)]
                        if this_hour.empty == False:
                            hour_isnull = this_hour[col].isnull().all()
                            if hour_isnull == True:
                                hour_tallies[hour]+=1
        for hour in range(0,24):
            print("Nulls for hour "+str(hour)+": "+str(hour_tallies[hour])+"\n")
        hour_tallies = pd.Series(hour_tallies)
        this_aqs_null_tallies[col] = hour_tallies
        col_newname = str(aqs_code)+"-"+col
        all_site[col_newname] = hour_tallies
        #print(pd.Series(hour_tallies))
        #this_aqs_null_tallies
        #print("Hours with no data: ")
    this_aqs_null_tallies.to_csv(descr_path+aqs_code+"-hour_nulls-2000.csv", index=False)
all_site.to_csv(descr_path+'all_sites_null_hour_tallies_2000.csv', index = False)


# #Just doing hours first
# #This does hours as COLUMNS
# all_site_hour_null_tallies = pd.DataFrame()
# for aqs_code in list(data['AQS_Code'].unique()):
#     site_records = data[data["AQS_Code"] == aqs_code]
#     print("Data for site: "+str(aqs_code)+ '\n')
#     this_aqs_null_tallies = pd.DataFrame()
#     for col in ['no', 'no2', 'nox', 'o3']:
#         #pol_tallies = pd.DataFrame()
#         hour_tallies = {}
#         print("Pollutant: "+col+"\n")
#         for month in range(1,13):
#             #fill out
#             for day in range(1,32):
#                 #fill out 
#                 for hour in range(0,24):
#                     if hour not in hour_tallies: 
#                         hour_tallies[hour] = 0
#                     else:
#                         this_hour = site_records[(site_records['day']==day) & (site_records['hour']==hour) & (site_records['month']==month)]
#                         if this_hour.empty == False:
#                             hour_isnull = this_hour[col].isnull().all()
#                             if hour_isnull == True:
#                                 hour_tallies[hour]+=1
#         for hour in range(0,24):
#             print("Nulls for hour "+str(hour)+": "+str(hour_tallies[hour])+"\n")
#         this_aqs_null_tallies[col] = pd.Series(hour_tallies)
#         #print(pd.Series(hour_tallies))
#         #this_aqs_null_tallies
#         #print("Hours with no data: ")
#     this_aqs_null_tallies.to_csv(descr_path+aqs_code+"-hour_nulls-2000.csv", index=False)  
