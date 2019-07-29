#!/usr/bin/env python
# coding: utf-8

# In[4]:


import plac
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from matplotlib import pyplot
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler

# One question is, is the distribution of missing data different than the distribution of data that is present? So is it even valid to do imputation on the missing data when it could be differently shaped due to good reasons? 

@plac.annotations(
    input_path=("Path containing the data files to ingest", "option", "p", str),
    input_prefix=("{$prefix}year.csv", "option", "P", str),
    input_suffix=("year{$suffix}.csv", "option", "S", str),
    output_path=("Path to write the resulting numpy sequences / transform cache", "option", "o", str),
    year_begin=("First year to process", "option", "b", int),
    year_end=("Year to stop with", "option", "e", int),
    fillgps=("Add correct GPS information because it is often missing in Data_structure_3", "flag", "G"),
    naninvalid=("Set invalid col entries to nan", "flag", "N"),
    dropnan=("Drop nan rows", "flag", "D"),
    masknan=("Mask nan rows", "option", "M", float),
    fillnan=("Fill nan rows", "option", "F", float),
    aqsnumerical=("Convert AQS code to numerical", "flag", "A"),
    houston=("Only run for Houston sites", "flag", "H"),
    chunksize=("Process this many records at one time", "option", 'C', int)
)
def main(input_path: str = '/project/lindner/air-pollution/level3_data/',
         input_prefix: str = "Data_",
         input_suffix: str = "",
         output_path: str = '/project/lindner/air-pollution/current/2019/data-formatted/houston',
         year_begin: int = 2000,
         year_end: int = 2018,
         fillgps: bool = False,
         naninvalid: bool = False,
         dropnan: bool = False,
         masknan: float = None,
         fillnan: float = None,
         aqsnumerical: bool = False,
         houston: bool = False,
         chunksize: int = 200000):
    data1 = pd.read_csv("/project/lindner/air-pollution/current/2019/data-formatted/concat_aqs/Transformed_Data_48_201_0695.csv")
    data2 = pd.read_csv("/project/lindner/air-pollution/current/2019/data-formatted/concat_aqs/Transformed_Data_48_201_0416.csv")
    #Goal is to impute Park Place o3 from all other features
    y = data2['o3']
    data1 = data1.add_prefix('MoodyTowers_')
    data2 = data2.drop(['o3'], axis = 'columns').add_prefix('ParkPlace_')
    #Because of unneeded columns leftover from faulty script
    data1X = data1.replace('48_201_0695', 0)
    data2X = data2.replace('48_201_0416', 1)
    X = pd.concat([data1X,data2X], ignore_index=True)
    #X, y = X.dropna(), y.dropna()
    X = X.dropna(how = 'all', axis = 'columns')
    X, y = np.array(X), np.array(y)
    scaler = MinMaxScaler()
    X = BiScaler().fit_transform(X)
    X = SoftImpute().fit_transform(X)
    y = BiScaler().fit_transform(y)
    y = SoftImpute().fit_transform(y)
    #X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    # Initialising the RNN
    regressor = Sequential()
    #Layers
    regressor.add(Dense(25, input_dim=21, activation='relu', kernel_initializer='he_uniform'))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(25, input_dim=21, activation='relu', kernel_initializer='he_uniform'))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(25, input_dim=21, activation='relu', kernel_initializer='he_uniform'))
    regressor.add(Dropout(0.2))
    # Adding the output layer
    regressor.add(Dense(units = 1))

    # Compiling the RNN
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # Fitting the RNN to the Training set
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, verbose=1)
    # evaluate the model
    train_mse = model.evaluate(X_train, y_train, verbose=0)
    test_mse = model.evaluate(X_test, y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))
    # plot loss during training
    pyplot.title('Loss / Mean Squared Error')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
    pyplot.savefig(output_path+"MSE_of_LSTM_model.png")
    regressor.save(output_path+"model.h5")


# In[ ]:


if __name__ == '__main__':
    plac.call(main)

