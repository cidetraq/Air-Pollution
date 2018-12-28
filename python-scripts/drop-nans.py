
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np


# In[ ]:


def transform(year):
    filename='Data_'+str(year)+'_ready.csv'
    data=pd.read_csv(filename)
    data=data.dropna()
    data.to_csv('Data_'+str(year)+'_ready.csv', index=False)


# In[7]:


years=np.arange(2000,2014)
for year in years:
    transform(year)

