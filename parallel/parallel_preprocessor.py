import pandas as pd
import numpy as np
import multiprocessing
import plac
import os
import d
from processor_pipeline import WindowFunction, SequenceBuilder, SequenceFeatureEnricher
from time import sleep


class SiteProcessor(multiprocessing.Process):
    def __init__(self, site):

        self.window_function = WindowFunction(window_size=d.WINDOW_STRIDE)
        self.sequence_builder = SequenceBuilder(sequence_length=d.SEQUENCE_LENGTH,
                                                prediction_window=d.PREDICTION_WINDOW,
                                                prediction_names=d.OUTPUT_COLUMNS)
        self.sequence_feature_enricher = SequenceFeatureEnricher(regression_features=d.REGRESSION_FEATURES,
                                                                 std_features=d.STD_FEATURES)

        self.finished_event = multiprocessing.Event()
        self.wakeup_event = multiprocessing.Event()
        self.data_event = multiprocessing.Event()
        self.idle_event = multiprocessing.Event()
        self.cmd_event = multiprocessing.Event()

        parent_data_pipe, child_data_pipe = multiprocessing.Pipe()
        parent_cmd_pipe, child_cmd_pipe = multiprocessing.Pipe()
        parent_response_pipe, child_response_pipe = multiprocessing.Pipe()

        self.data_pipe = parent_data_pipe
        self.cmd_pipe = parent_cmd_pipe
        self.response_pipe = parent_response_pipe

        multiprocessing.Process.__init__(self, target=self._procedure,
                                         args=[site, child_data_pipe, child_cmd_pipe, child_response_pipe])

    def _procedure(self, args):
        self.site = args[0][0]
        self.data_pipe = args[0][1]
        self.cmd_pipe = args[0][2]
        self.response_pipe = args[0][3]

        while not self.finished_event.is_set():

            self.idle_event.set()

            self.wakeup_event.wait()
            self.wakeup_event.clear()

            self.idle_event.clear()

            if self.cmd_event.is_set():
                self._process_cmd()
                self.cmd_event.clear()

            if self.data_event.is_set():
                self._process_data()
                self.data_event.clear()

            if self.finished_event.is_set():
                break

    def _process_data(self):
        nd = self.data_pipe.recv()
        nd = self.window_function.process(nd)
        nd = self.sequence_builder.process(nd)
        self.sequence_feature_enricher.process(nd)

    def _process_cmd(self):
        cmd, data = self.cmd_pipe.recv()
        if cmd == "get_minmax":
            self.response_pipe.send(self.window_function.minmax)
        elif cmd == "set_minmax":
            self.window_function.minmax = self.data_pipe.recv()
            self.response_pipe.send(None)
        elif cmd == "save_and_shutdown":
            self._save()
            self.finished_event.set()
            self.response_pipe.send(None)

    def _save(self):
        # TODO
        pass

    def is_idle(self):
        return self.idle_event.is_set()

    def finished(self):
        self.finished_event.set()
        self.wakeup_event.set()

    def givejob(self, job):
        self.response_pipe.send(job)
        self.data_event.set()
        self.wakeup_event.set()

    def cmd(self, cmd, data=None):
        self.cmd_pipe.send([cmd, data])
        self.cmd_event.set()
        self.wakeup_event.set()
        return self.response_pipe.recv()


class WorkerManager(object):
    def __init__(self, num_jobs):
        self.num_jobs = num_jobs
        self.workers = {}

    def __getitem__(self, key):
        return self.workers[key]

    def __iter__(self):
        return self.workers.keys()

    def addworker(self, site):
        self.workers[site] = SiteProcessor(site)
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
    ingest_path=("Path containing the data files to ingest", "option", "P", str),
    year_begin=("First year to process", "option", "b", int),
    year_end=("Year to stop with", "option", "e", int),
    num_jobs=("Number of workers simultameously ingesting data. Set to number of cores", "option", "J", int)
)
def main(ingest_path: str = '/some/default/path/here', num_jobs: int = 8, year_begin: int = 2000, year_end: int = 2018):
    workers = WorkerManager(num_jobs=num_jobs)

    for year in range(year_begin, year_end):
        df = pd.read_csv(os.path.join(ingest_path, "%d_mark.csv" % year))

        for site in d.SITES:

            job = df[df['site'] == site].values()

            if len(job) == 0:
                continue

            if site not in workers:
                workers.addworker(site)

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

        for i in d.NUM_INPUTS:
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


if __name__ == '__init__':
    plac.call(main)
