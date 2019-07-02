# In[3]:


import pandas as pd
import numpy as np
from mpi4py import MPI
import plac
import os
import sys

HOUSTON = {'48_201_0051': {'Longitude': -95.474167, 'Latitude': 29.623889},
           '48_201_0558': {'Longitude': -95.3536111, 'Latitude': 29.5894444},
           '48_201_0572': {'Longitude': -95.105, 'Latitude': 29.583333},
           '48_201_0551': {'Longitude': -95.1602778, 'Latitude': 29.8586111},
           '48_201_6000': {'Longitude': -95.2535982, 'Latitude': 29.6843603},
           '48_201_0669': {'Longitude': -95.252778, 'Latitude': 29.694722},
           '48_201_0695': {'Longitude': -95.3414, 'Latitude': 29.7176},
           '48_201_0307': {'Longitude': -95.2599093, 'Latitude': 29.718799},
           '48_201_0670': {'Longitude': -95.257222, 'Latitude': 29.701944},
           '48_201_0673': {'Longitude': -95.256697, 'Latitude': 29.7023},
           '48_201_0671': {'Longitude': -95.255, 'Latitude': 29.706111},
           '48_201_0069': {'Longitude': -95.2611301, 'Latitude': 29.7062492},
           '48_201_1035': {'Longitude': -95.2575931, 'Latitude': 29.7337263},
           '48_201_0057': {'Longitude': -95.238469, 'Latitude': 29.734231},
           '48_201_1049': {'Longitude': -95.2224669, 'Latitude': 29.716611},
           '48_201_0803': {'Longitude': -95.1785379, 'Latitude': 29.7647877},
           '48_201_1034': {'Longitude': -95.2205822, 'Latitude': 29.7679965},
           '48_201_1052': {'Longitude': -95.38769, 'Latitude': 29.81453},
           '48_201_0024': {'Longitude': -95.3261373, 'Latitude': 29.9010364}}

# In[ ]:


def transform(df: pd.DataFrame, year: int, fillgps: bool = False, naninvalid: bool = False, dropnan: bool = False, masknan: float = None, fillnan: float = None, aqsnumerical: bool = False, site = []) -> pd.DataFrame:

    if aqsnumerical:
        df['AQS_Code'].str.replace('_', '')
        df['AQS_Code'] = df['AQS_Code'].astype(int)

    return df


def run_job(job: dict):

    if job['cmd'] == 'transform':
        output_df = pd.DataFrame()    
        for year_idx, year in enumerate(range(job['year_begin'], job['year_end'])):
            path = job['input_path']+"Transformed_Data_"+str(year)+"_"+job['site']
            print(path)
            try: 
                aqs_year = pd.read_csv(path)
            except BaseException:
                continue
            output_df = pd.concat([output_df, aqs_year])
        output_df.to_csv(job['output_path'])                                
                          
        print("Saved job "+str(job['site'])+"to disk.")
        return


@plac.annotations(
    input_path=("Path containing the data files to ingest", "option", "p", str),
    input_prefix=("{$prefix}year.csv", "option", "P", str),
    input_suffix=("year{$suffix}.csv", "option", "S", str),
    output_path=("Path to write the resulting numpy sequences / transform cache", "option", "o", str),
    year_begin=("First year to process", "option", "b", int),
    year_end=("Year to stop with", "option", "e", int),
    fillgps=("Add correct GPS information because it is often missing in Data_structure_3", "flag", "G"),
    naninvalid=("Set invalid col entries to nan", "flag", "N"),
    dropnan=("Drop nan rows", "flag", "D"),
    masknan=("Mask nan rows", "option", "M", float),
    fillnan=("Fill nan rows", "option", "F", float),
    aqsnumerical=("Convert AQS code to numerical", "flag", "A"),
    houston=("Only run for Houston site", "flag", "H"),
    chunksize=("Process this many records at one time", "option", 'C', int)
)
def main(input_path: str = '/project/lindner/air-pollution/current/2019/data-formatted/mpi-test/',
         input_prefix: str = "Data_",
         input_suffix: str = "",
         output_path: str = '/project/lindner/air-pollution/current/2019/data-formatted/concat_aqs/',
         year_begin: int = 2000,
         year_end: int = 2018,
         fillgps: bool = False,
         naninvalid: bool = False,
         dropnan: bool = False,
         masknan: float = None,
         fillnan: float = None,
         aqsnumerical: bool = False,
         houston: bool = False,
         chunksize: int = 200000):

    if masknan is not None and fillnan is not None:
        sys.exit("Error: fillnan and masknan cannot both be set.")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_procs = comm.Get_size()

    print("(mpi4py) rank: %d n_procs: %d" % (rank, n_procs))
    sys.stdout.flush()

    if rank == 0:
        
        # Create jobs
        jobs = []

        outstanding_jobs = 0
        n_proc = 1
        
        all_files = os.listdir(input_path)
        #String processing to get all unique aqs
        unique_aqs = []
        for file in all_files: 
            spl = file.split("_")
            aqs = "_".join(spl[3:])
            unique_aqs.append(aqs)
        for aqs in unique_aqs:
            transform_name = "Transformed_Data_"+str(aqs)
            job = {'cmd': 'transform',
                   'site': str(aqs),
                   'aqsnumerical': aqsnumerical,
                   'input_path': input_path,
                   'year_begin': year_begin,
                   'year_end': year_end,
                   'output_path': os.path.join(output_path, transform_name)}
            jobs.append(job)

        n_proc = 1
                
        # Distribute one full round robin of jobs
        while len(jobs) > 0:
            comm.isend(jobs.pop(), dest=n_proc, tag=1)

            n_proc += 1
            outstanding_jobs += 1

            if n_proc == n_procs:
                break

        # Distribute more jobs as workers become free
        while outstanding_jobs > 0:
            req = comm.irecv(tag=2)
            data = req.wait()

            outstanding_jobs -= 1

            if len(jobs) > 0:
                comm.isend(jobs.pop(), dest=data['rank'], tag=1)
                outstanding_jobs += 1

            print("%d jobs left." % (len(jobs) + outstanding_jobs))
            sys.stdout.flush()

        # Clean up
        for nproc in range(1, n_procs):
            req = comm.isend({'cmd': 'shutdown'}, nproc, tag=1)
            req.wait()

        print("Node %d shutting down." % rank)
        sys.stdout.flush()

    else:
        while True:
            req = comm.irecv(source=0, tag=1)
            job = req.wait()
            print(job['cmd'])

            if job['cmd'] == 'transform':

                print("Got job: %s" % job['site'])
                sys.stdout.flush()                
                run_job(job)

                print("Finished job: %s" % job['site'])
                sys.stdout.flush()

                result = {'year': job['site'], 'rank': rank}
                req = comm.isend(result, dest=0, tag=2)
                req.wait()

            elif job['cmd'] == 'shutdown':
                print("Node %d shutting down." % rank)
                sys.stdout.flush()
                return


if __name__ == '__main__':
    plac.call(main)
# In[ ]:




