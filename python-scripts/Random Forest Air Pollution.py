
# coding: utf-8

# In[54]:


import pandas as pd


# In[55]:


data=pd.read_csv('data00_100000_clean.csv')


# In[56]:


data=data.drop(['AQS_Code', 'year'], axis=1)


# In[57]:


data=data.drop(['date'],axis=1)


# In[58]:


data


# In[59]:


def time_to_num(time):
    splits=time.split(':')
    num=int(splits[0])*60+int(splits[1])
    return(int(num))


# In[60]:


time_to_num('06:15:00')


# In[61]:


data['time']=data['time'].apply(time_to_num)


# In[62]:


data['time']


# In[66]:


data=data.drop(['time'], axis=1)


# In[79]:


data_strs=data
data_strs['day']=data_strs['day'].apply(lambda x: str(x))
data_strs['month']=data_strs['month'].apply(lambda x: str(x))
data_strs['hour']=data_strs['hour'].apply(lambda x: str(x))


# In[80]:


dummies=pd.get_dummies(data_strs)


# In[82]:


dummies.columns


# In[69]:


data


# In[72]:


from keras.utils import np_utils


# In[77]:


day_cat=np_utils.to_categorical(data['day'])b
month_cat=np_utils.to_categorical(data['month'])
hour_cat=np_utils.to_categorical(data['hour'])


# In[83]:


# Use numpy to convert to arrays
import numpy as np
# Labels are the values we want to predict
labels = np.array(dummies['o3'])
# Remove the labels from the features
# axis 1 refers to the columns
features= dummies.drop('o3', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)


# In[84]:


# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 1)


# <h3><i>Establish baseline?

# <h2>Model Implementation Time</h2>

# In[86]:


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
#use verbose=1
# Train the model on training data
rf.fit(train_features, train_labels);


# In[89]:


import pickle

pickle.dump(rf, open('rf_1.pkl', 'wb'))


# In[91]:


from sklearn.externals import joblib

# Save to file in the current working directory
joblib_file = "rf_1.pkl"  
joblib.dump(rf, joblib_file)


# In[94]:


rf.score(test_features, test_labels)


# In[96]:


from sklearn import model_selection
kfold = model_selection.KFold(n_splits=10, random_state=0)
scoring = 'neg_mean_absolute_error'
results = model_selection.cross_val_score(rf, test_features, test_labels, cv=kfold, scoring=scoring)
results


# In[99]:


results.mean()


# <i>R-squared score on 1000 trees: 0.9516668132093452
#     
# Mean Absolute Error scores (cross-val): array([-4.19007583, -4.15650348, -4.06492095, -3.97509023, -4.09298393,
#        -4.21514153, -4.32852868, -4.11073779, -4.144749  , -3.97688133])
#        mean MAE: -4.1255612746321315

# <h3>Notes</h3>
# 
# Try with lower tree count, changing the maximal depth

# In[101]:


rf.n_features_


# In[104]:


feat_importances=rf.feature_importances_


# In[107]:


feature_importances = pd.DataFrame(feat_importances,
                                   index = dummies.drop('o3', axis=1).columns,
                                    columns=['importance']).sort_values('importance', ascending=False)


# In[108]:


feature_importances


# In[113]:


from sklearn.grid_search import GridSearchCV

rf_reg=RandomForestRegressor()
parameters= {'n_estimators':[100], 'max_features':[2,4,8,16,32,64,76], 'min_samples_leaf': [1,10,50,100,200,500], 'min_samples_split': [2,4,8,16]}
rf_grid = GridSearchCV(rf_reg, parameters, cv = 4)
rf_grid.fit(train_features, train_labels)


# In[116]:


rf_100=rf_grid.best_estimator_


# In[119]:


rf_100.score(test_features, test_labels)


# In[120]:


# Save to file in the current working directory
joblib_file = "rf_100.pkl"  
joblib.dump(rf_100, joblib_file)


# In[121]:


rf_10=RandomForestRegressor(n_estimators=10)
rf_10.fit(train_features, train_labels)


# In[124]:


rf_10.score(test_features, test_labels)


# In[125]:


joblib_file = "rf_10.pkl"  
joblib.dump(rf_10, joblib_file)


# In[114]:


rf_grid.best_params_


# In[126]:


# Use the forest's predict method on the test data
predictions = rf_10.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2))


# # Load from file
# joblib_model = joblib.load(joblib_file)
# 
# # Calculate the accuracy and predictions
# score = joblib_model.score(Xtest, Ytest)  
# print("Test score: {0:.2f} %".format(100 * score))  
# Ypredict = pickle_model.predict(Xtest)  
