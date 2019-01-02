import pandas as pd
import numpy as np
import multiprocessing
import plac
import os
import d
from processor_pipeline import WindowFunction, SequenceBuilder, SequenceFeatureEnricher
from time import sleep


class SiteProcessor(multiprocessing.Process):
    def __init__(self, site, output_path):

        self.window_function = WindowFunction(window_size=d.WINDOW_STRIDE)
        self.sequence_builder = SequenceBuilder(sequence_length=d.SEQUENCE_LENGTH,
                                                prediction_window=d.PREDICTION_WINDOW,
                                                prediction_names=d.OUTPUT_COLUMNS)
        self.sequence_feature_enricher = SequenceFeatureEnricher(regression_features=d.REGRESSION_FEATURES,
                                                                 std_features=d.STD_FEATURES)

        self.idle_event = multiprocessing.Event()

        self.cmd_queue = multiprocessing.Queue()
        self.response_queue = multiprocessing.Queue()

        self.site = site
        self.output_path = output_path

        multiprocessing.Process.__init__(self, target=self._procedure)

    def _procedure(self):

        while True:

            self.idle_event.set()

            cmd, data = self.cmd_queue.get()
            if cmd == "process_data":
                self._process_data(data)
            if cmd == "get_minmax":
                self.response_queue.put(self.window_function.minmax)
            elif cmd == "set_minmax":
                self.window_function.minmax = data
                self.response_queue.put(None)
            elif cmd == "save_and_shutdown":
                self._save()
                self.response_queue.put(None)
                self.idle_event.set()
                return

            self.idle_event.clear()

    def _process_data(self, nd):
        nd = self.window_function.process(nd)

        nd = self.sequence_builder.process(nd)

        if nd is not None:
            self.sequence_feature_enricher.process(nd)

    def _save(self):
        minmax = self.window_function.minmax

        if len(self.sequence_feature_enricher.sample_sequences) == 0:
            return

        # Convert to numpy arrays and scale the values
        sample_sequences = np.array(self.sequence_feature_enricher.sample_sequences)

        for i in range(0, d.NUM_INPUTS):

            # Bias to start at zero
            bias = -minmax[i][0]
            sample_sequences[:, :, i] += bias

            # Scale maximum value to 0
            scale = np.abs(minmax[i][1] - minmax[i][0])

            if scale != 0:
                sample_sequences[:, :, i] /= scale
            else:
                # If everything is the same, set everything to 0.
                sample_sequences[:, :, i] = 0

        labels = np.array(self.sequence_builder.labels)
        label_scaler_map = self.sequence_builder.labels_scaler_map

        for o in range(0, d.NUM_OUTPUTS):
            # Bias to start at zero
            labels[:, o] -= minmax[label_scaler_map[o]][0]

            # Scale maximum value to 0
            labels[:, o] /= np.abs(minmax[label_scaler_map[o]][0] - minmax[label_scaler_map[o]][1])

        sequence_features = np.array(self.sequence_feature_enricher.sequence_features)
        sequence_features_scalar_map = self.sequence_feature_enricher.sequence_features_scalar_map

        for f, s in enumerate(sequence_features_scalar_map):
            # Bias to start at zero
            sequence_features[:, f] -= minmax[s][0]

            # Scale maximum value to 0
            sequence_features[:, f] /= np.abs(minmax[s][0] - minmax[s][1])

        # np.save(os.path.join(self.output_path, self.site + '_labels.nd'), labels)

    def is_idle(self):
        return self.idle_event.is_set()

    def givejob(self, job):
        self.cmd_queue.put(['process_data', job])

    def cmd(self, cmd, data=None):
        self.cmd_queue.put([cmd, data])

        return self.response_queue.get()


class WorkerManager(object):
    def __init__(self, num_jobs):
        self.num_jobs = num_jobs
        self.workers = {}

    def __getitem__(self, key):
        return self.workers[key]

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n == len(self.workers):
            raise StopIteration

        self.n += 1

        return [i for i in self.workers.keys()][self.n-1]

    def addworker(self, site: str, output_path: str):
        self.workers[site] = SiteProcessor(site, output_path)
        self.workers[site].start()

    def wait(self):
        # Wait for jobs to finish
        while True:
            jobs = 0

            for site in self.workers:
                jobs += 1 if not self.workers[site].idle_event.is_set() else 0

            if jobs < self.num_jobs:
                break

            sleep(0.1)


def transform(df, year):

    df['wind_x_dir'] = df['windspd'] * np.cos(df['winddir'] * (np.pi / 180))
    df['wind_y_dir'] = df['windspd'] * np.sin(df['winddir'] * (np.pi / 180))
    df['hour'] = pd.to_datetime(df['epoch'], unit='s').dt.hour

    if year < 2014:
        df = df[df['nox_flag'] == 'VAL']
        df = df[df['no_flag'] == 'VAL']
        df = df[df['o3_flag'] == "VAL"]
        df = df[df['temp_flag'] == "VAL"]
    if year >= 2014:
        df = df[df['nox_flag'] == 'VAL']
        df = df[df['no_flag'] == 'VAL']
        df = df[df['o3_flag'] == "K"]
        df = df[df['temp_flag'] == "K"]
    df = df[~df['winddir'].isna()]
    df = df[~df['AQS_Code'].isna()]

    df = df.drop(
        ['co_flag', 'humid', 'humid_flag', 'pm25', 'pm25_flag', 'so2', 'so2_flag', 'solar', 'solar_flag', 'dew',
         'dew_flag', 'redraw', 'co', 'no_flag', 'no2_flag', 'nox_flag', 'o3_flag', 'winddir_flag', 'windspd_flag',
         'temp_flag'], axis=1)
    return df


@plac.annotations(
    ingest_path=("Path containing the data files to ingest", "option", "P", str),
    ingest_prefix=("{$prefix}year.csv", "option", "p", str),
    ingest_suffix=("year{$suffix}.csv", "option", "s", str),
    output_path=("Path to write the resulting numpy sequences", "option", "o", str),
    year_begin=("First year to process", "option", "b", int),
    year_end=("Year to stop with", "option", "e", int),
    num_jobs=("Number of workers simultameously ingesting data. Set to number of cores", "option", "J", int),
    reset_cache=("Do not use existing cache files (rebuild new ones)", "flag")
)
def main(ingest_path: str = '/some/default/path/here/input',
         ingest_prefix: str = "",
         ingest_suffix: str = "",
         output_path: str = '/some/default/path/here/output',
         num_jobs: int = 1, year_begin: int = 2000, year_end: int = 2018, reset_cache=False):
    workers = WorkerManager(num_jobs=num_jobs)

    for year_idx, year in enumerate(range(year_begin, year_end)):

        input_name = "%s%d%s.csv" % (ingest_prefix, year, ingest_suffix)
        cache_name = input_name + '.cache'

        print("Processing: %s" % input_name)

        if not reset_cache:
            try:
                df = pd.read_csv(os.path.join(ingest_path, cache_name))
                print("Loaded %s from cache" % input_name)
            except FileNotFoundError:
                df = pd.read_csv(os.path.join(ingest_path, input_name))
                df = transform(df, year)
                df.to_csv(os.path.join(ingest_path, cache_name))
        else:
            df = pd.read_csv(os.path.join(ingest_path, input_name))
            df = transform(df, year)
            df.to_csv(os.path.join(ingest_path, cache_name))

        for site in df['AQS_Code'].unique():

            print("Processing(%s): %s" % (year, site))

            job = df[df['AQS_Code'] == site][d.INPUT_COLUMNS].values

            if len(job) == 0:
                continue

            if site not in workers:
                workers.addworker(site, output_path)

            workers[site].givejob(job)
            workers.wait()

        # Coordinate min/max between all workers
        minmaxrows = []
        for site in d.SITES:
            if site not in workers:
                continue
            minmaxrows.append(workers[site].cmd("get_minmax"))

        minmaxrows = np.array(minmaxrows)
        minmax = np.zeros((d.NUM_INPUTS, 2))

        for i in range(0, d.NUM_INPUTS):
            minmax[i][0] = np.min(minmaxrows[:, i])
            minmax[i][1] = np.max(minmaxrows[:, i])

        for site in d.SITES:
            if site not in workers:
                continue

            workers[site].cmd("set_minmax", minmax)

        # Save and shutdown
        for site in d.SITES:
            if site not in workers:
                continue

            workers[site].cmd("save_and_shutdown")
            workers.wait()

        for site in d.SITES:
            if site not in workers:
                continue
            print("Joining: %s" % site)
            workers[site].join()


if __name__ == '__main__':
    plac.call(main)
