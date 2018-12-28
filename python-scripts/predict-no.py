
# coding: utf-8

# In[1]:


from sklearn.externals import joblib


# In[ ]:


joblib_file = "2000-2013-rf-no.pkl"  
model = joblib.load(joblib_file)

import pandas as pd

data14=pd.read_csv('Data_2014_pre_no.csv')
data15=pd.read_csv('Data_2015_pre_no.csv')
data16=pd.read_csv('Data_2016_pre_no.csv')
data17=pd.read_csv('Data_2017_pre_no.csv')
data=pd.DataFrame()
for year in [data14,data15,data16,data17]:
    data=pd.concat([data,year])

data_strs=data
data_strs['day']=data_strs['day'].apply(lambda x: str(x))
data_strs['month']=data_strs['month'].apply(lambda x: str(x))
data_strs['hour']=data_strs['hour'].apply(lambda x: str(x))
data_strs['year']=data_strs['hour'].apply(lambda x: str(x))
dummies=pd.get_dummies(data_strs)

# Use numpy to convert to arrays
import numpy as np
predictions=model.predict(np.array(data))
import pickle
pickle.dump(predictions, open('rf-no-predictions-14-17.pkl', 'wb'))

