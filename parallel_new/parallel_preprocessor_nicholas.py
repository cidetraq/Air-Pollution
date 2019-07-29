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
        #we need all the site-specific code to stay in windowing test
        #deleted SequenceBuilder and SequenceFeatureEnricher objects
        self.idle_event = multiprocessing.Event()

        self.cmd_queue = multiprocessing.Queue()
        self.response_queue = multiprocessing.Queue()

        self.site = site
        self.index = index
        self.output_path = output_path

        self.masknan = masknan
        self.nd = None
        multiprocessing.Process.__init__(self, target=self._procedure)

    def _procedure(self):

        while True:

            self.idle_event.set()
            cmd, data = self.cmd_queue.get()
            self.idle_event.clear()

            if cmd == "process_data":
                self.nd = self._process_data(data)
            #Deleted minmax
            elif cmd == "save_and_shutdown":
                self._save()
                self.response_queue.put(None)
                self.idle_event.set()
                return

    def _process_data(self, nd):
        nd = self.window_function.process(nd)
        
        #np.save(os.path.join(self.output_path, "window_avgs"), nd)
        return nd 
    def _save(self):
        #deleted everything except saving windowed data
        def _save(nd: np.ndarray, name: str):
            out_name = "window_avgs_"+str(self.site)+".npy"
            out_desc_name = 'window_avgs_'+str(self.site)+".desc"
            
            # (Nicholas) nd comes from earlier process_data
            np.save(os.path.join(self.output_path, out_name), nd)

            description = "%s\n%s" % (str(nd.dtype), str(nd.shape))
            open(os.path.join(self.output_path, out_desc_name), "w").write(description)
            
        _save(self.nd, "nd")

    def is_idle(self):
        return self.idle_event.is_set()

    def givejob(self, job):
        self.cmd_queue.put(['process_data', job])

    def cmd(self, cmd, data=None):
        self.cmd_queue.put([cmd, data])

        return self.response_queue.get()

    #keep workermanager intact
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
        # (Nicholas) Don't really understand this- does it get number of jobs each loop?
        # Wait for jobs to finish
        while True:
            jobs = 0

            for site in self.workers:
                jobs += 1 if not self.workers[site].idle_event.is_set() else 0

            if jobs < self.num_jobs:
                break

            sleep(0.1)

#delete long-unused transform_file

#Leaving many of these because it doesn't hurt to keep them.
@plac.annotations(
    ingest_path=("Path containing the data files to ingest", "option", "P", str),
    ingest_prefix=("{$prefix}year.csv", "option", "p", str),
    ingest_suffix=("year{$suffix}.csv", "option", "s", str),
    output_path=("Path to write the resulting numpy sequences / transform cache", "option", "o", str),
    year_begin=("First year to process", "option", "b", int),
    year_end=("Year to stop with (every year below this and in range will run)", "option", "e", int),
    num_jobs=("Number of workers simultameously ingesting data. Set to number of cores", "option", "J", int),
    reset_cache=("Do not use existing cache files (rebuild new ones)", "flag"),
    site=("Only run a specific site", "option", "S", str),
    masknan=("Value that missing values will be replaced with", "option", "M", float),
    testing_path=("Just run on this file only", "option", "t", str)
)
def main(ingest_path: str = '/project/lindner/moving/summer2018/Data_structure_3/',
         ingest_prefix: str = "Transformed_Data_",
         ingest_suffix: str = "",
         output_path: str = '/project/lindner/air-pollution/current/2019/data-formatted/hourly/',
         num_jobs: int = 1,
         year_begin: int = 2000,
         year_end: int = 2018,
         reset_cache: bool = False,
         site: str = None,
         masknan: float = None,
         testing_path: str = None):
    workers = WorkerManager(num_jobs=num_jobs)

    site_index = 0

    jobs = []
   
    if testing_path:
        df = pd.read_csv(testing_path)
        if site is None:
            site = list(df['AQS_Code'].unique())[0]
        workers.addworker(site, site_index, output_path, masknan=masknan)
        job = df[d.INPUT_COLUMNS].values
        workers[site].givejob(job)
        print("New job received for site "+str(site))
        workers.wait()
   
    #delete minmax here

    # Save and shutdown
    for site in d.SITES:
        if site not in workers:
            continue
        print("saving and shutting down")
        workers[site].cmd("save_and_shutdown")
        workers.wait()

    for site in d.SITES:
        if site not in workers:
            continue
        print("Joining: %s" % site)
        workers[site].join()


if __name__ == '__main__':
    plac.call(main)