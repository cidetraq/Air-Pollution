
# coding: utf-8

# In[ ]:


#%matplotlib inline
#import matplotlib
#import matplotlib.pyplot as plt
import pandas
import numpy as np
from collections import deque
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from keras.models import Model, load_model
from keras import backend as K
from keras.layers import Dense, LSTM, Input, Flatten
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard


# In[ ]:


profiles={'cluster': {'in_path': '/project/lindner/moving/summer2018/2019/data-intermediate/', 'out_path': '/project/lindner/moving/summer2018/2019/models/'},
         'nicholas': {'in_path': 'D:/programming-no-gdrive/air-pollution/data-intermediate/', 'out_path': 'D:/programming-no-gdrive/DASH/Air Pollution/models/'}}


# In[ ]:


def main(user, window_stride, nd_window,load, epochs=10):
    #Notebook
    os.chdir('../python-scripts')
    from format_data_labels import create_data_labels

    # Average window_stride elements together to form a single row
    sample_hours = window_stride / 12.0
    print("Sample Hours: %f" % sample_hours)
    
    # Number of future samples to mean for prediction
    prediction_window = int(24 / sample_hours)
    print("Prediction Window: %d" % prediction_window)

    # Length of the windowed sequence
    sequence_length = int(7*24 / sample_hours)
    print("Sequence Length: %d" % sequence_length)

    # Number of things we are doing regression to predict
    num_outputs = 4
    in_path=profiles[user]['in_path']
    out_path=profiles[user]['out_path']
    nd_window=pickle.load(open(in_path+nd_window, 'rb'))
    data, labels=create_data_labels(window_stride, nd_window)
    print('Created data and labels')
    # Number of features we take from the data
    print('Data shape: '+str(data.shape))
    input_features = data.shape[2]
    print('Sequence length: '+str(sequence_length))
    print('Number of input features: '+str(input_features))
    num_features = input_features
    num_inputs = input_features
    model_params=[sequence_length, input_features]
    model=create_model(data, labels, out_path, model_params, num_outputs, load, epochs, window_stride)


# #Theano
# class AttLayer(Layer):
#     def __init__(self, **kwargs):
#         self.init = initializations.get('normal')
#         #self.input_spec = [InputSpec(ndim=3)]
#         super(AttLayer, self).__init__(** kwargs)
# 
#     def build(self, input_shape):
#         assert len(input_shape)==3
#         #self.W = self.init((input_shape[-1],1))
#         self.W = self.init((input_shape[-1],))
#         #self.input_spec = [InputSpec(shape=input_shape)]
#         self.trainable_weights = [self.W]
#         super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!
# 
#     def call(self, x, mask=None):
#         eij = K.tanh(K.dot(x, self.W))
# 
#         ai = K.exp(eij)
#         weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')
# 
#         weighted_input = x*weights.dimshuffle(0,1,'x')
#         return weighted_input.sum(axis=1)
# 
#     def get_output_shape_for(self, input_shape):
#         return (input_shape[0], input_shape[-1])

# In[ ]:


def model_architecture(sequence_length, input_features, num_outputs, r2, window_stride):
# For some reason putting some extra dimensions before an LSTM works wonders
    layer_input = Input(shape=(sequence_length, input_features), name='inputs')
    dense_1 = Dense(128, input_dim=(sequence_length, input_features))(layer_input)
    layer_lstm = LSTM(64, return_sequences=True, dropout=0.5)(dense_1)
    layer_flatten = Flatten()(layer_lstm)

    layer_output = Dense(num_outputs, activation='linear', name='outputs')(layer_flatten)

    model = Model(inputs=[layer_input], outputs=[layer_output])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[r2])
    print(model.summary())
    return model


# In[ ]:


def create_model(data, labels, out_path, model_params,num_outputs,load,window_stride,epochs=10):
    def r2(y_true, y_pred):
        SS_res =  K.sum(K.square(y_true - y_pred)) 
        SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
        return ( 1 - SS_res/(SS_tot + K.epsilon()) )
    sequence_length=model_params[0]
    input_features=model_params[1]

    def sched(epoch, lr):
        new_lr = 0.001 * (0.95 ** epoch)
        print("Epoch(%d) LR: %f" % (epoch+1, new_lr))
        return new_lr

    lr_decay = LearningRateScheduler(schedule=sched) 

    filepath='lstm_w'+str(window_stride)+'_f'+str(input_features)+'_o'+str(num_outputs)+'.h5'

    checkpoint = ModelCheckpoint(out_path+filepath, monitor='val_r2', verbose=1, save_best_only=True, mode='max')

    tensorboard = TensorBoard(log_dir='./tb', histogram_freq=0, batch_size=128, write_graph=True, write_grads=False)
    if load==None:
        model=model_architecture(sequence_length, input_features, num_outputs, r2, window_stride)
    else: 
        model=load_model(out_path+load, custom_objects={'r2': r2})
    model.fit(x=data, y=labels, batch_size=128, epochs=epochs, validation_split=0.2, verbose=True, callbacks=[lr_decay, checkpoint, tensorboard])
    return model


# In[ ]:


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", '--user', type=str,
                        help="cluster, nicholas, carroll")
    #parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2],
    #                    help="increase output verbosity")
    parser.add_argument('-w', '--window_stride', type=int, help='window stride')
    parser.add_argument('-f', '--filename', type=str, help='filename of nd_window')
    parser.add_argument('-l', '--load_model', type=str, help='filename of model to load. if blank, will create new model.')
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs to train')
    args = parser.parse_args()
    main(args.user, args.window_stride, args.filename, args.load_model, args.epochs)


# In[ ]:


#main('nicholas', 12, 'windowed_2000.pkl')


# 
# #Plotting
# model.load_weights("weights.best.hdf5")
# 
# plt.rcParams['figure.figsize'] = (20, 10)
# plt.rcParams['font.size'] = 16
# 
# for seq in range(0, data.shape[0] - sequence_length):
# 
#     lookup = {'no': (0, 0), 'no2':(0, 1), 'nox':(1, 0), 'o3':(1, 1)}
# 
#     pred = model.predict(data[seq].reshape(1, sequence_length, num_features))[0]
#     fig, ax = plt.subplots(2, 2)
# 
#     for idx,f in enumerate([(4, 'no'), (5, 'no2'), (6, 'nox'), (7, 'o3')]):
# 
#         feature_index, feature_name = f
# 
#         X = []
#         Y_actual = []
# 
#         for i in range(0, sequence_length + int(24*(1/sample_hours))):
#             X.append(seq+i)
#             Y_actual.append(data[seq+i][-1][feature_index])
# 
#         Y_actual = np.array(Y_actual)
# 
#         predicted_mean = pred[feature_index - 4]
#         actual_mean = np.mean(Y_actual[sequence_length:])
#         rolling_mean = np.mean(Y_actual[:sequence_length])
#         rolling_std = np.std(Y_actual[:sequence_length])
# 
#         Y_pred = Y_actual.copy()
#         Y_pred[sequence_length:] = predicted_mean
#         Y_pred[:sequence_length] = np.nan
# 
#         Y_actual_mean = Y_actual.copy()
#         Y_actual_mean[sequence_length:] = actual_mean
#         Y_actual_mean[:sequence_length] = np.nan
# 
#         Y_rolling_mean = Y_actual.copy()
#         Y_rolling_mean[:sequence_length] = rolling_mean
#         Y_rolling_mean[sequence_length:] = np.nan
# 
#         Y_rolling_std_upper = Y_actual.copy()
#         Y_rolling_std_upper[:sequence_length] = rolling_mean + rolling_std
#         Y_rolling_std_upper[sequence_length:] = np.nan
# 
#         Y_rolling_std_lower = Y_actual.copy()
#         Y_rolling_std_lower[:sequence_length] = rolling_mean - rolling_std
#         Y_rolling_std_lower[sequence_length:] = np.nan   
# 
#         subplot = ax[lookup[feature_name][0]][lookup[feature_name][1]]
# 
#         subplot.plot(X, Y_actual, color='black', linewidth=4.0)
#         subplot.plot(X, Y_actual_mean, color='green', linewidth=4.0)
#         subplot.plot(X, Y_pred, color='purple', linewidth=4.0)
#         subplot.plot(X, Y_rolling_mean, color='green', linewidth=4.0)
#         subplot.plot(X, Y_rolling_std_upper, color='orange', linewidth=4.0)
#         subplot.plot(X, Y_rolling_std_lower, color='orange', linewidth=4.0)
# 
#         subplot.grid()
# 
#         subplot.set_title("%s 24 hour mean prediction" % (feature_name,))
# 
#         subplot.set_xlabel("Hours")
#         subplot.set_ylabel("Scaled Concentration")
# 
#     fig.legend(['Actual Continuous', 'Actual Mean', 'Predicted Mean', 'Rolling Mean', 'Standard Deviation'])
#     fig.tight_layout()
# 
#     plt.savefig('charts/%.05d.png' % seq)
#     #plt.show()
#     plt.close()
# 
#     print("Rendered %d" % seq)
