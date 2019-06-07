import numpy as np
import pandas as pd
from collections import deque
from typing import List, Optional


class NumpyPipelineUnit(object):
    def pipe(self, input: np.ndarray) -> Optional[np.ndarray]:
        raise Exception("abstract class function called")

    def __init__(self, phase_rise: int, phase_fall: int):
        self.phase_rise = phase_rise
        self.phase_fall = phase_fall


class PandasPipelineUnit(object):
    def pipe(self, input: pd.DataFrame) -> Optional[pd.DataFrame]:
        raise Exception("abstract class function called")


class SequencerUnit(NumpyPipelineUnit):
    def __init__(self, length: int):
        NumpyPipelineUnit.__init__(self, phase_rise = length, phase_fall = 0)

        self.sequence = deque(maxlen=length)
        self.length = length
        self.leftovers = None

    def pipe(self, input: np.ndarray) -> Optional[np.ndarray]:
        if self.leftovers is not None:
            input = np.concatenate([self.leftovers, input])

        num_sequences = input.shape[0] + 1 - self.length - (lf.length - len(self.sequence))

        if num_sequences <= 0:
            return None

        # Allocate memory for the generated sequences
        sequences = np.zeros((num_sequences, self.length, input.shape[1]))

        for idx in enumerate(0, input.shape[0]):

            self.sequence.append(input[idx])

            if len(self.sequence) == self.length:
                sequences[idx] = np.concatenate(self.sequence)

        return sequences


class WindowUnit(NumpyPipelineUnit):
    def __init__(self, length: int, features: List):
        NumpyPipelineUnit.__init__(self, phase_rise = length, phase_fall = 0)
        
        self.length = length
        self.leftovers = None
        self.features = features

    def pipe(self, input: np.ndarray) -> Optional[np.ndarray]:
        if self.leftovers is not None:
            input = np.concatenate([self.leftovers, input])

            w_start = 0
            w_end = self.length

            # Allocate memory for the windows
            windows = np.zeros((input.shape[0] - self.length + 1, input.shape[1]))

            while w_end < input.shape[0]:
                for idx, feature in enumerate(self.features):
                    fname, ftype = feature

                    if ftype == "disrete":
                        # Use the midpoint for discrete windows
                        windows[w_start, idx] = input[, int((w_start + w_end)/2)]
                    elif ftype == "continuous":
                        # Use the mean for continuous windows
                        windows[w_start, idx] = np.nanmean(input[, idx])

                # Slide window
                w_start += 1
                w_end += 1

            # Store leftovers
            self.leftovers = input[w_start:]

            return windows


class SuzieUnit(NumpyPipelineUnit):
    def __init__(self):
        pass
    def pipe(self, input: np.ndarray) -> np.ndarray:
        pass


class CSVPreprocessorUnit(PandasPipelineUnit):
    def __init__(self, year: int, dropnan: bool = False, masknan: float = None, fillnan: float = None, sites = []):
        self.sites = sites
        self.year = year
        self.dropnan = dropnan
        self.masknan = masknan
        self.fillnan = fillnan
        self.aqsnumerical = aqsnumerical
        self.sites = sites

    def pipe(self, input: pd.DataFrame) -> pd.DataFrame:

        if len(self.sites) > 0:
            input.drop(input[~input['AQS_Code'].isin(list(self.sites.keys()))].index, inplace=True)

        if self.dropnan:
            if year < 2014:
                val = 'VAL'
            if year >= 2014:
                val = 'K'

            input.dropna(inplace=True)

        input['wind_x_dir'] = input['windspd'] * np.cos(input['winddir'] * (np.pi / 180))
        input['wind_y_dir'] = input['windspd'] * np.sin(input['winddir'] * (np.pi / 180))
        input['hour'] = pd.to_datetime(input['epoch'], unit='s').dt.hour
        input['day_of_year'] = pd.Series(pd.to_datetime(input['epoch'], unit='s'))
        input['day_of_year'] = input['day_of_year'].dt.dayofyear

        return input
