#!/usr/bin/env python
# coding: utf-8

# In[1]:


profiles={'cluster': {'in_path': '/project/lindner/moving/summer2018/2019/data-intermediate/', 'out_path': '/project/lindner/moving/summer2018/2019/data-intermediate/'}}


# In[ ]:


def create_data_labels(window_stride,nd_window):
    import numpy as np
    from collections import deque
    import pickle
    # Average window_stride elements together to form a single row

    sample_hours = window_stride / 12.0
    print("Sample Hours: %f" % sample_hours)

    # Number of future samples to mean for prediction
    prediction_window = int(24 / sample_hours)
    print("Prediction Window: %d" % prediction_window)

    # Length of the windowed sequence
    sequence_length = int(7*24 / sample_hours)
    print("Sequence Length: %d" % sequence_length)

    # Number of features we take from the data
    input_features = 9
    num_features = input_features
    num_inputs = input_features

    # Number of things we are doing regression to predict
    num_outputs = 4

    # Create sequences
    data = []
    labels = []


    rows = deque(maxlen=sequence_length)

    for idx, r in enumerate(nd_window):

        rows.append([a for a in r])

        # We need the entire sequence filled to make a prediction about the future mean
        if len(rows) < sequence_length:
            continue

        # Since we are predicting the mean, make sure we do not go out of bounds in the future
        if idx+1 + prediction_window > nd_window.shape[0]:
            break

        data.append(rows.copy())

        # We are predicting the future mean values
        u_24_no = np.mean( nd_window[idx+1 : idx+1 + prediction_window, 4] )
        u_24_no2 = np.mean( nd_window[idx+1 : idx+1 + prediction_window, 5] )
        u_24_nox = np.mean( nd_window[idx+1 : idx+1 + prediction_window, 6] )
        u_24_o3 = np.mean( nd_window[idx+1 : idx+1 + prediction_window, 7] )

        labels.append([u_24_no, u_24_no2, u_24_nox, u_24_o3])

    data = np.array(data)
    labels = np.array(labels)
    return data,labels 


# In[ ]:


def main(user, window_stride, filename):
    in_path=profiles[user]['in_path']
    out_path=profiles[user]['out_path']
    nd_window=np.load(open(in_path+filename, 'rb'))
    data, labels=create_data_labels(window_stride,nd_window)
    pickle.dump(data, open(out_path+filename+'_data.ndarray', 'wb'), protocol=4)
    pickle.dump(labels, open(out_path+filename+'_labels.ndarray', 'wb'), protocol=4)


# In[ ]:


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", '--user', type=str,
                        help="cluster, nicholas, carroll")
    parser.add_argument('-w', '--window', type=int, help='window size')
    parser.add_argument('-f', '--filename', type=str, help='filename')
    args = parser.parse_args()
    main(args.user, args.window, args.filename)

