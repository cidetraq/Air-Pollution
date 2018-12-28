
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[4]:


#Local
#data=pd.read_csv('d00.csv')
data=pd.read_csv('Data_2000_ready.csv')


# In[8]:


data.columns


# In[6]:


data=data.dropna()
data=data.drop(['no2','nox'], axis=1)


# In[25]:


data_strs=data
data_strs['day']=data_strs['day'].apply(lambda x: str(x))
data_strs['month']=data_strs['month'].apply(lambda x: str(x))
data_strs['hour']=data_strs['hour'].apply(lambda x: str(x))
data_strs['year']=data_strs['hour'].apply(lambda x: str(x))


# In[ ]:


dummies=pd.get_dummies(data_strs)


# In[ ]:


# Use numpy to convert to arrays
import numpy as np
# Labels are the values we want to predict
labels = np.array(dummies['no'])
# Remove the labels from the features
# axis 1 refers to the columns
features= dummies.drop('no', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)


# In[ ]:


# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 1)


# <h2>Model Implementation Time</h2>

# In[ ]:


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 10, random_state = 0, verbose=1)
#use verbose=1
# Train the model on training data
rf.fit(train_features, train_labels)


# In[ ]:


from sklearn.externals import joblib

# Save to file in the current working directory
joblib_file = "rf-no-2000.pkl"  
joblib.dump(rf, joblib_file)


# In[ ]:


print(rf.score(test_features, test_labels))


# In[ ]:


from sklearn import model_selection
kfold = model_selection.KFold(n_splits=4, random_state=0)
scoring = 'neg_mean_absolute_error'
results = model_selection.cross_val_score(rf, test_features, test_labels, cv=kfold, scoring=scoring)
print('Mean absolute errors: '+str(results))


# In[ ]:


print(results.mean())


# In[ ]:


feat_importances=rf.feature_importances_


# In[ ]:


feature_importances = pd.DataFrame(feat_importances,
                                   index = dummies.drop('o3', axis=1).columns,
                                    columns=['importance']).sort_values('importance', ascending=False)


# In[ ]:


feature_importances.to_csv('feat-importances-no.csv')


# from sklearn.grid_search import GridSearchCV
# 
# rf_reg=RandomForestRegressor()
# parameters= {'n_estimators':[100], 'max_features':[2,4,8,16,32,64,76], 'min_samples_leaf': [1,10,50,100,200,500], 'min_samples_split': [2,4,8,16]}
# rf_grid = GridSearchCV(rf_reg, parameters, cv = 4)
# rf_grid.fit(train_features, train_labels)

# rf_100=rf_grid.best_estimator_

# rf_100.score(test_features, test_labels)

# # Save to file in the current working directory
# joblib_file = "rf_100.pkl"  
# joblib.dump(rf_100, joblib_file)

# rf_grid.best_params_

# # Use the forest's predict method on the test data
# predictions = rf_10.predict(test_features)
# # Calculate the absolute errors
# errors = abs(predictions - test_labels)
# # Print out the mean absolute error (mae)
# print('Mean Absolute Error:', round(np.mean(errors), 2))

# # Load from file
# joblib_model = joblib.load(joblib_file)
# 
# # Calculate the accuracy and predictions
# score = joblib_model.score(Xtest, Ytest)  
# print("Test score: {0:.2f} %".format(100 * score))  
# Ypredict = pickle_model.predict(Xtest)  
