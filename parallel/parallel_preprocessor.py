import pandas as pd
import numpy as np
import multiprocessing
import plac
import os
import d
from processor_pipeline import WindowFunction, SequenceBuilder, SequenceFeatureEnricher
from time import sleep


class SiteProcessor(multiprocessing.Process):
    def __init__(self, site: str, index: int, output_path: str, masknan: float):

        self.window_function = WindowFunction(window_size=d.WINDOW_STRIDE, masknan=masknan)
        self.sequence_builder = SequenceBuilder(sequence_length=d.SEQUENCE_LENGTH,
                                                prediction_window=d.PREDICTION_WINDOW,
                                                prediction_names=d.OUTPUT_COLUMNS,
                                                masknan=masknan)
        self.sequence_feature_enricher = SequenceFeatureEnricher(regression_features=d.REGRESSION_FEATURES,
                                                                 std_features=d.STD_FEATURES, masknan=masknan)

        self.idle_event = multiprocessing.Event()

        self.cmd_queue = multiprocessing.Queue()
        self.response_queue = multiprocessing.Queue()

        self.site = site
        self.index = index
        self.output_path = output_path

        self.masknan = masknan

        multiprocessing.Process.__init__(self, target=self._procedure)

    def _procedure(self):

        while True:

            self.idle_event.set()
            cmd, data = self.cmd_queue.get()
            self.idle_event.clear()

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

        def _save(nd: np.ndarray, name: str):
            out_name = '%0.3d_%s' % (self.index, name)
            out_desc_name = '%0.3d_%s.desc' % (self.index, name)

            np.save(os.path.join(self.output_path, out_name), nd)

            description = "%s\n%s" % (str(nd.dtype), str(nd.shape))
            open(os.path.join(self.output_path, out_desc_name), "w").write(description)

        # Add longitude and lattitude
        latlong_features = np.zeros((sequence_features.shape[0], 2))
        latlong_features[:, 0] = (d.SITES[self.site]['Latitude'] + 90.) / 180.
        latlong_features[:, 1] = (d.SITES[self.site]['Longitude'] + 180.) / 360.

        if self.masknan is not None:
            labels[labels == np.nan] = self.masknan
            sample_sequences[sample_sequences == np.nan] = self.masknan
            sequence_features[sequence_features == np.nan] = self.masknan

        _save(labels, "labels")
        _save(sample_sequences, "sequences")
        _save(sequence_features, "sequence_features")
        _save(latlong_features, "latlong_features")

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

        return [i for i in self.workers.keys()][self.n - 1]

    def addworker(self, site: str, index: int, output_path: str, masknan: float = None):
        self.workers[site] = SiteProcessor(site, index, output_path, masknan=masknan)
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


@plac.annotations(
    input_path=("Path containing the data files to ingest", "option", "P", str),
    input_prefix=("{$prefix}year.csv", "option", "p", str),
    input_suffix=("year{$suffix}.csv", "option", "s", str),
    output_path=("Path to write the resulting numpy sequences / transform cache", "option", "o", str),
    year_begin=("First year to process", "option", "b", int),
    year_end=("Year to stop with", "option", "e", int),
    num_jobs=("Number of workers simultameously ingesting data. Set to number of cores", "option", "J", int),
    reset_cache=("Do not use existing cache files (rebuild new ones)", "flag"),
    site=("Only run a specific site", "option", "S", str),
    masknan=("Mask nan rows instead of dropping them", "option", "M", float)
)
def main(input_path: str = '/project/lindner/moving/summer2018/Data_structure_3',
         input_prefix: str = "Data_",
         input_suffix: str = "",
         output_path: str = '/project/lindner/moving/summer2018/2019/data-formatted/parallel',
         num_jobs: int = 8,
         year_begin: int = 2000,
         year_end: int = 2018,
         site: str = None,
         masknan: float = None):
    workers = WorkerManager(num_jobs=num_jobs)

    site_index = 0

    for year_idx, year in enumerate(range(year_begin, year_end)):

        input_name = "%s%d%s.csv" % (input_prefix, year, input_suffix)
        transform_name = 'Transformed_' + input_name

        print("Processing: %s" % input_name)

        df = pd.read_csv(os.path.join(input_path, transform_name))

        if site is None:
            sites = df['AQS_Code'].unique()
        else:
            sites = [site]

        for site in sites:

            print("Processing(%s): %s" % (year, site))

            job = df[df['AQS_Code'] == site][d.INPUT_COLUMNS].values

            if len(job) == 0:
                continue

            if site not in workers:
                workers.addworker(site, site_index, output_path, masknan=masknan)
                site_index += 1

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
