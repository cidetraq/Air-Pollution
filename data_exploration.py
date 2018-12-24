
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data=pd.read_csv('data00_100000.csv')


# In[17]:


data['nox'].isna().value_counts()


# In[24]:


data['nox_flag'].value_counts()


# In[14]:


data.columns


# In[12]:


#data=data.drop(['redraw', 'co'],axis=1)


# In[23]:


data=data[data['nox_flag']=='VAL']


# In[27]:


len(data)


# In[28]:


data['no_flag'].value_counts()


# In[29]:


data=data[data['no_flag']=='VAL']


# In[30]:


data['no2_flag'].value_counts()


# In[31]:


data['o3_flag'].value_counts()


# In[32]:


data=data[data['o3_flag']=="VAL"]


# In[33]:


len(data)


# In[34]:


data['pm25_flag'].value_counts()


# In[37]:


data['pm25']


# In[38]:


data['so2_flag'].value_counts()


# In[39]:


data['so2']


# In[40]:


data['solar']


# In[41]:


data['temp'].isna().value_counts()


# In[43]:


data['temp_flag'].value_counts()


# In[44]:


data=data[data['temp_flag']=="VAL"]


# In[46]:


data['winddir_flag'].value_counts()


# In[58]:


data[data['winddir'].isna()]


# In[59]:


data=data[~data['winddir'].isna()]


# In[62]:


len(data)


# In[64]:


data['windspd_flag'].value_counts()


# In[65]:


data['dew'].isna().value_counts()


# In[67]:


data=data.drop(['co_flag', 'humid', 'humid_flag', 'pm25', 'pm25_flag', 'so2', 'so2_flag', 'solar',  'solar_flag', 'temp', 'temp_flag', 'dew', 'dew_flag'], axis=1)


# In[72]:


#dropping now unnecessary flags
data=data.drop(['no_flag', 'no2_flag', 'nox_flag', 'o3_flag', 'winddir_flag', 'windspd_flag'], axis=1)


# In[73]:


data.columns


# In[74]:


data['time'] = pd.to_datetime(data['epoch'], unit='s').dt.time
data['date'] = pd.to_datetime(data['epoch'], unit='s').dt.date
data['day'] = pd.to_datetime(data['epoch'], unit='s').dt.day
data['month'] = pd.to_datetime(data['epoch'], unit='s').dt.month
data['year'] = pd.to_datetime(data['epoch'], unit='s').dt.year
data['hour']  = pd.to_datetime(data['epoch'], unit='s').dt.hour


# In[78]:


data


# In[80]:


import numpy as np
data['wind_x_dir'] = data['windspd'] * np.cos(data['winddir']*(np.pi/180))
data['wind_y_dir'] = data['windspd'] * np.sin(data['winddir']*(np.pi/180))


# In[81]:


data


# In[82]:


data.to_csv('data00_100000_clean.csv', index=False)


# In[83]:


#For this data, delete AWS_Code column and year column

