from collections import deque
import numpy as np
import d
from typing import List, Optional


class WindowFunction(object):

    def __init__(self, window_size: int):
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
        while sample < nd.shape[0] - self.window_size:
            for i in range(0, d.NUM_INPUTS):

                window_value = np.mean(nd[sample:sample + self.window_size, i])

                if window_value < self.minmax[i][0]:
                    self.minmax[i][0] = window_value
                elif window_value > self.minmax[i][1]:
                    self.minmax[i][1] = window_value

                windows[int(sample / self.window_size)][i] = window_value

            sample += self.window_size

        # Save leftovers for next window
        self.leftovers = nd[-self.window_size:]

        return windows


class SequenceBuilder(object):
    def __init__(self, sequence_length: int, prediction_window: int, prediction_names: List[str], statistics: bool):

        self.sequence = deque(maxlen=sequence_length)
        self.sequence_length = sequence_length
        self.prediction_window = prediction_window
        self.prediction_names = prediction_names
        self.statistics = statistics

        if statistics:
            self.gaps = {}

        self.leftovers = None

        self.labels = []

        # So we can map label features back to scaling values
        self.labels_scaler_map = []
        for o in range(0, d.NUM_OUTPUTS):
            self.labels_scaler_map.append(d.INPUT_MAP[d.OUTPUT_COLUMNS[o]])

    def process(self, nd: np.ndarray) -> Optional[np.ndarray]:

        if self.leftovers is not None:
            nd = np.concatenate([self.leftovers, nd])

        num_return_sequences = nd.shape[0] + len(self.sequence) - (
                self.sequence_length - 1) - self.prediction_window

        if num_return_sequences <= 0:
            self.leftovers = nd
            return None

        sequences = np.zeros((num_return_sequences, self.sequence_length, nd.shape[1]))
        sequences_idx = 0

        for sample in range(0, nd.shape[0]):
            self.sequence.append(nd[sample])

            if self.statistics:
                if len(self.sequence) >= 2:
                    key = self.sequence[-1][d.INPUT_MAP['epoch']] - self.sequence[-2][d.INPUT_MAP['epoch']]
                    if key in self.gaps:
                        self.gaps[key] += 1
                    else:
                        self.gaps[key] = 1

            # Wait to have a complete sequence
            if len(self.sequence) < self.sequence_length:
                continue

            # Since we are predicting the mean, make sure we do not go out of bounds in the future
            if sample + 1 + self.prediction_window > nd.shape[0]:
                break

            nd_sequence = np.array(self.sequence)
            sequences[sequences_idx] = nd_sequence

            # Labels
            predictions = []
            for name in self.prediction_names:
                prediction = np.mean(nd[sample + 1: sample + 1 + self.prediction_window, d.INPUT_MAP[name]])
                predictions.append(prediction)

            self.labels.append(predictions)

            sequences_idx += 1

        self.leftovers = nd[-self.prediction_window:]

        return sequences


class SequenceFeatureEnricher(object):
    def __init__(self, regression_features=True, std_features=True):
        self.regression_features = regression_features
        self.std_features = std_features

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

    def process(self, nd: np.ndarray):

        # Add some features
        for sequence in range(0, nd.shape[0]):

            features_to_add = []

            if self.regression_features:
                for f in range(d.ENRICH_START, d.NUM_INPUTS):

                    m = np.sum(nd[sequence][:, f]) / np.sum(np.arange(0, nd.shape[1]))
                    b = nd[sequence][:, f][0]

                    features_to_add.extend([m, b])

            if self.std_features:
                for f in range(d.ENRICH_START, d.NUM_INPUTS):
                    features_to_add.append(np.std(nd[sequence][:, f]))

            self.sample_sequences.append(nd[sequence])
            self.sequence_features.append(features_to_add)

