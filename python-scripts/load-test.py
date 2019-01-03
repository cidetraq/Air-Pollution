#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.models import load_model

def r2(y_true, y_pred):
        SS_res =  K.sum(K.square(y_true - y_pred)) 
        SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
        return ( 1 - SS_res/(SS_tot + K.epsilon()) )

model=load_model('D:/programming-no-gdrive/DASH/Air Pollution/weights.best.hdf5')
model.predict()

