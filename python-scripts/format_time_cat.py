#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


def time_cat(data):
    data['month']=data['month'].apply(lambda x: str(x))
    data['hour']=data['hour'].apply(lambda x: str(x))
    data['day']=data['day'].apply(lambda x: str(x))
    pre_dummies=pd.concat([data['hour'], data['day'], data['month']], axis=1)
    dummies=pd.get_dummies(pre_dummies)
    data=pd.concat([data, dummies], axis=1)
    data=data.drop_duplicates()
    data=data.drop(['hour', 'day', 'month'], axis=1)
    return data

