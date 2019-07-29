from collections import deque
import numpy as np
import d
from typing import List, Optional


class WindowFunction(object):

    def __init__(self, window_size: int, masknan: float = None):
        self.masknan = masknan
        self.window_size = window_size
        self.leftovers = None

        # Keep track of min and max for each feature
        self.minmax = np.zeros((d.NUM_INPUTS, 2))
        self.minmax[:, 0] = np.inf
        self.minmax[:, 1] = -np.inf

    def process(self, nd: np.ndarray):
        if self.leftovers is not None:
            nd = np.concatenate([self.leftovers, nd])

        windows = np.zeros((int(nd.shape[0] / self.window_size), nd.shape[1]))

        sample = 0
        #while sample less than number of rows minus window size
        while sample <= nd.shape[0] - self.window_size:
            #select each input column ( features )
            for i in range(0, d.NUM_INPUTS):
                #subset is part of array in selected column with rows from current sample to sample plus window size
                subset = nd[sample:sample + self.window_size, i]
                num_nans = np.isnan(subset).sum()
                #acceptable less or equal to 25% missing data
                if num_nans <= d.WINDOW_STRIDE*.25:
                    #set window value to the mean of subset excluding nans
                    window_value = np.nanmean(nd[sample:sample + self.window_size, i])

                    if window_value < self.minmax[i][0]:
                        self.minmax[i][0] = window_value
                    elif window_value > self.minmax[i][1]:
                        self.minmax[i][1] = window_value

                #unacceptable more than 25% missing data
                else: 
                    print("Over 25% nans")
                    #set to 0 masking value
                    window_value = 0
                #the next value in selected column of windows is window value
                windows[int(sample / self.window_size)][i] = window_value
            #increment sample by window size (12 more)
            sample += self.window_size

        # Save leftovers for next window (where does it go?)
        self.leftovers = nd[sample:]
        print("Length of windows: "+str(windows.shape[0]))
        return windows


class SequenceBuilder(object):
    def __init__(self, sequence_length: int, prediction_window: int, prediction_names: List[str],
                 masknan: float = None):

        self.sequence = deque(maxlen=sequence_length)
        self.sequence_length = sequence_length
        self.prediction_window = prediction_window
        self.prediction_names = prediction_names
        self.masknan = masknan

        self.leftovers = None

        self.labels = []

        # So we can map label features back to scaling values
        self.labels_scaler_map = []
        for o in range(0, d.NUM_OUTPUTS):
            self.labels_scaler_map.append(d.INPUT_MAP[d.OUTPUT_COLUMNS[o]])

    def process(self, nd: np.ndarray) -> Optional[np.ndarray]:

        if self.leftovers is not None:
            nd = np.concatenate([self.leftovers, nd])

        have_samples = nd.shape[0] + len(self.sequence)
        seq_pred_length = self.sequence_length + self.prediction_window
        num_return_sequences = have_samples - (seq_pred_length + 1)
        print("Num return sequences: "+str(num_return_sequences))
        #Where does nd come from, since leftovers was already set to None? (Nicholas)
        if num_return_sequences <= 0:
            self.leftovers = nd
            return None

        # TODO: dynamic number of sequences depending on proportion of NaN in sequence and prediction windows
        sequences = np.zeros((num_return_sequences, self.sequence_length, nd.shape[1]))

        sample = 0
        while sample < num_return_sequences + (self.sequence_length - len(self.sequence)):

            self.sequence.append(nd[sample])

            # Wait to have a complete sequence
            # Nicholas: How does this act as a gate?
            if len(self.sequence) < self.sequence_length:
                continue

            nd_sequence = np.array(self.sequence)
            sequences[sample] = nd_sequence
            # Now the sequences array has one more sequence of complete length
            
            # Labels
            """
            predictions = []
            for name in self.prediction_names:
                prediction = np.nanmean(nd[sample + 1: sample + 1 + self.prediction_window, d.INPUT_MAP[name]])
                predictions.append(prediction)

            self.labels.append(predictions)
            
            """
            sublabels = []
            for name in self.prediction_names:
                label = nd[sample+self.sequence_length, d.INPUT_MAP[name]]
                sublabels.append(label)
            self.labels.append(sublabels)
            sample += 1


        self.leftovers = nd[sample:]

        return sequences


class SequenceFeatureEnricher(object):
    def __init__(self, regression_features=True, std_features=True, kmeans_features=True, masknan: float = None, n_clusters=8):
        self.regression_features = regression_features
        self.std_features = std_features
        self.kmeans_features= kmeans_features
        self.n_clusters=n_clusters
        self.masknan = masknan
        self.sample_sequences = []
        self.sequence_features = []
        
        # So we can map sequence features back to minmax values for scaling
        self.sequence_features_scalar_map = []
        if regression_features:
            for f in range(d.ENRICH_START, d.NUM_INPUTS):
                self.sequence_features_scalar_map.append(f)
                self.sequence_features_scalar_map.append(f)
        if std_features:
            for f in range(d.ENRICH_START, d.NUM_INPUTS):
                self.sequence_features_scalar_map.append(f)
                
    def kmeans_process(self, nd: np.ndarray):
        from sklearn.cluster import KMeans
        kmeans=KMeans(n_clusters=self.n_clusters, n_jobs=-1, verbose=1).fit(nd)
        return kmeans.labels_, kmeans.cluster_centers_
    
    def process(self, nd: np.ndarray):

        # Add some features
        for sequence in range(0, nd.shape[0]):

            features_to_add = []

            if self.regression_features:
                for f in range(d.ENRICH_START, d.NUM_INPUTS):
                    m = np.nansum(nd[sequence][:, f]) / np.nansum(np.arange(0, nd.shape[1]))
                    b = nd[sequence][:, f][0]

                    features_to_add.extend([m, b])

            if self.std_features:
                for f in range(d.ENRICH_START, d.NUM_INPUTS):
                    features_to_add.append(np.nanstd(nd[sequence][:, f]))
           
            self.sample_sequences.append(nd[sequence])
            self.sequence_features.append(features_to_add)
            
    '''if self.kmeans_features:
        labels, cluster_centers=self.kmeans_process(nd)
        self.sequence_features.extend([labels,                             cluster_centers])'''
                

'''
   so, basically what happens is you start with the rows sequences
so (row, sequence_index, feature_index)
then you reshape
(row, sequence_index*feature_index)
then you concatenate on axis=1
np.concatenate([sequences_reshaped, my_new_features], axis=1)      
     
'''