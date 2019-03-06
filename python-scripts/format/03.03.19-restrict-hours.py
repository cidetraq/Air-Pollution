#!/usr/bin/env python
# coding: utf-8

# <h1>Objective</h1>
# Reformat data structure 3 data into hourly averages, dropping averages with less than 75% of rows present in a given hour, replace nans with avgs or skip entirely.
# -> This may be the cause of the failure of my recent code which found that there were only 2 sites with a day's worth of data each. This code took data with pre-dropped rows from ds3, with five minute intervals

# In[3]:


import pandas as pd


# <h2>Make sure to change paths before uploading to cluster

# In[4]:


#on cluster
out_path = "/project/lindner/moving/summer2018/2019/data-formatted/03.03.19-restrict-hours/"
data = pd.read_csv('/project/lindner/moving/summer2018/Data_structure_3/Data_2000.csv')
#on local
#out_path = "D:/programming-no-gdrive/air-pollution/data-formatted/03.03.19-restrict-hours/"
#data = pd.read_csv('D:/programming-no-gdrive/air-pollution/data-sample/100k/data00_100000.csv')


# <h2>See data columns</h2>

# In[ ]:


#data.columns


# <h2>Output all AQS Codes (suspect some bad data)

# with open(descr_path+'codes.txt', 'w') as file:
#     for aqs in data['AQS_Code'].unique():
#         file.write(str(aqs)+"\n")

# <h2>Drop unneeded columns</h2>

# In[5]:


data=data.drop(['redraw', 'co', 'co_flag', 'humid', 'humid_flag', 'pm25', 'pm25_flag', 'so2', 'so2_flag', 'solar', 'solar_flag', 'temp', 'temp_flag', 'dew', 'dew_flag', 'winddir_flag', 'windspd_flag', 'no_flag', 'no2_flag', 'nox_flag', 'o3_flag'], axis=1)


# <h2>Add time features (Hour, Day, Month). Code from format-mark in notebooks/format

# In[6]:


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


# <h2>Main processing

# In[53]:


print("Working with Chinmay's 71 \"valid\" sites with at least one data point per month on selected columns.")
print("Note: Usually winddir and windspd are both present or absent at once so we select for just winddir.")
all_site = pd.DataFrame()
for aqs_code in list(data['AQS_Code'].unique()):
    #We evaluate for all columns remaining except for Latitude/ Longitude as those may be missing
    site_records = data[data["AQS_Code"] == aqs_code].drop(['Latitude', 'Longitude'], axis = 1)
    print("Data for site: "+str(aqs_code))
    hourly_site = pd.DataFrame()
    for month in range(1,13):
        #fill out
        for day in range(1,32):
            #fill out 
            for hour in range(0,24):
                this_hour = site_records[(site_records['day']==day) & (site_records['hour']==hour) & (site_records['month']==month)]
                nan_hour = this_hour.isnull().sum()
                nanover75 = False
                for nan_col_hour in nan_hour:
                    if nan_col_hour > 9:
                        nanover75 = True
                        break
                if nanover75 == False:
                    #This hour has at least 75% non-missing data
                    mean_row = this_hour.mean(axis=0, numeric_only=True)
                    
                    """ 
                    
                    means = {}
                    cols = list(this_hour.columns)
                    cols.remove("AQS_Code")
                    for col in cols:
                    means[col] = mean_row[col]
                    newrow = this_hour.fillna(value = means).drop(['epoch'], axis = 1)
                    
                    """
                    
                    mean_row = mean_row.to_frame().transpose().drop(['epoch'], axis = 1)
                    hourly_site = pd.concat([hourly_site, mean_row])
    hourly_site = hourly_site.dropna()
    hourly_site.to_csv(out_path+aqs_code+"-hourly.csv", index = False)
    hourly_site['AQS_Code'] = aqs_code
    all_site = pd.concat([all_site, hourly_site])
all_site.to_csv(out_path+"all_sites-2000.csv", index = False)           

