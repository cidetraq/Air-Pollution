import pandas as pd
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

        self.data_pip = parent_data_pipe
        self.write_pipe = parent_cmd_pipe
        self.read_pipe = parent_response_pipe

        multiprocessing.Process.__init__(self, target=self._procedure,
                                         args=[site, child_data_pipe, child_cmd_pipe, child_response_pipe,
                                               self.wakeup_event, self.data_event, self.idle_event, self.finished_event,
                                               self.cmd_event])

    def _procedure(self, args):
        self.site = args[0][0]
        self.data_pipe = args[0][1]
        self.cmd_pipe = args[0][2]
        self.response_pipe = args[0][3]
        self.wakeup_event = args[0][4]
        self.data_event = args[0][5]
        self.idle_event = args[0][6]
        self.finished_event = args[0][7]
        self.cmd_event = args[0][8]

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

        self._save()

    def _process_data(self):
        # TODO
        pass

    def _process_cmd(self):
        # TODO
        pass

    def _save(self):
        # TODO
        pass

    def is_idle(self):
        return self.idle_event.is_set()

    def finished(self):
        self.finished_event.set()
        self.wakeup_event.set()

    def givejob(self, job):
        self.read_pipe.send(job)
        self.data_event.set()
        self.wakeup_event.set()


@plac.annotations(
    ingest_path=("Path containing the data files to ingest", "option", "P", str),
    year_begin=("First year to process", "option", "b", int),
    year_end=("Year to stop with", "option", "e", int),
    num_jobs=("Number of workers simultameously ingesting data. Set to number of cores", "option", "J", int)
)
def main(ingest_path: str = '/some/default/path/here', num_jobs: int = 8, year_begin: int = 2000, year_end: int = 2017):
    workers = {}

    for year in range(year_begin, year_end):
        df = pd.read_csv(os.path.join(ingest_path, "%d_mark.csv" % year))

        for site in d.SITES:

            job = df[df['site'] == site].values()

            if len(job) == 0:
                continue

            if site not in workers:
                workers[site] = SiteProcessor(site)
                workers[site].start()

            workers[site].givejob(job)

            # Wait for jobs to finish
            while True:
                jobs = 0

                for site in workers:
                    jobs += 1 if not workers[site].idle_event.is_set() else 0

                if jobs < num_jobs:
                    break

                sleep(0.1)


if __name__ == '__init__':
    plac.call(main)
