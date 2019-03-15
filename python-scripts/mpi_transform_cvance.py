import pandas as pd
import numpy as np
from mpi4py import MPI
import plac
import os
import sys

HOUSTON = ['48_201_0051', '48_201_0558', '48_201_0572', '48_201_0551', '48_201_6000', '48_201_0669', '48_201_0695', '48_201_0307', '48_201_0670', '48_201_0673', '48_201_0671', '48_201_0069', '48_201_1035', '48_201_0057', '48_201_1049', '48_201_0803', '48_201_1034', '48_201_1052', '48_201_0024']

def transform(df: pd.DataFrame, year: int, masknan: float = None, fillnan: float = None, sites = []) -> pd.DataFrame:
    df = df[df['AQS_Code'].isin(sites)]

    if masknan is None and fillnan is None:

        if year < 2014:
            df = df[df['nox_flag'] == 'VAL']
            df = df[df['no_flag'] == 'VAL']
            df = df[df['no2_flag'] == 'VAL']
            df = df[df['o3_flag'] == "VAL"]
            df = df[df['temp_flag'] == "VAL"]
        if year >= 2014:
            df = df[df['nox_flag'] == 'K']
            df = df[df['no_flag'] == 'K']
            df = df[df['o3_flag'] == 'K']
            df = df[df['temp_flag'] == 'K']

        df = df[~df['winddir'].isna()]
        df = df[~df['AQS_Code'].isna()]

        df = df.drop(
            ['co_flag', 'humid', 'humid_flag', 'pm25', 'pm25_flag', 'so2', 'so2_flag', 'solar', 'solar_flag', 'dew',
             'dew_flag', 'redraw', 'co', 'no_flag', 'no2_flag', 'nox_flag', 'o3_flag', 'winddir_flag', 'windspd_flag',
             'temp_flag', 'Longitude', 'Latitude'], axis=1)

        df = df.dropna()

    df['wind_x_dir'] = df['windspd'] * np.cos(df['winddir'] * (np.pi / 180))
    df['wind_y_dir'] = df['windspd'] * np.sin(df['winddir'] * (np.pi / 180))
    df['hour'] = pd.to_datetime(df['epoch'], unit='s').dt.hour
    df['day_of_year'] = pd.Series(pd.to_datetime(df['epoch'], unit='s'))
    df['day_of_year'] = df['day_of_year'].dt.dayofyear

    if masknan is not None or fillnan is not None:
        df = df.drop(
            ['co_flag', 'humid', 'humid_flag', 'pm25', 'pm25_flag', 'so2', 'so2_flag', 'solar', 'solar_flag', 'dew',
             'dew_flag', 'redraw', 'co', 'no_flag', 'no2_flag', 'nox_flag', 'o3_flag', 'winddir_flag', 'windspd_flag',
             'temp_flag', 'Longitude', 'Latitude'], axis=1)
    if masknan is not None:
        df[df.isna()] = np.nan
    elif fillnan is not None:
        df = df.fillna(fillnan)

    return df


def run_job(job: dict):
    df = pd.read_csv(job['input_path'])
    df = transform(df, job['year'], job['masknan'], job['fillnan'], job['sites'])
    df.to_csv(job['output_path'])


@plac.annotations(
    input_path=("Path containing the data files to ingest", "option", "P", str),
    input_prefix=("{$prefix}year.csv", "option", "p", str),
    input_suffix=("year{$suffix}.csv", "option", "s", str),
    output_path=("Path to write the resulting numpy sequences / transform cache", "option", "o", str),
    year_begin=("First year to process", "option", "b", int),
    year_end=("Year to stop with", "option", "e", int),
    masknan=("Mask nan rows instead of dropping them", "option", "M", float),
    fillnan=("Mask nan rows instead of dropping them", "option", "F", float),
    houston=("Only run for Houston sites", "option", "H", bool),

)
def main(input_path: str = '/project/lindner/moving/summer2018/Data_structure_3',
         input_prefix: str = "Data_",
         input_suffix: str = "",
         output_path: str = '/project/lindner/moving/summer2018/2019/data-formatted/parallel',
         year_begin: int = 2000,
         year_end: int = 2018,
         masknan: float = None,
         fillnan: float = None,
         houston: bool = True):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_procs = comm.Get_size()

    print("rank: %d n_procs: %d" % (rank, n_procs))
    sys.stdout.flush()

    if rank == 0:

        n_proc = 1
        n_jobs = 0

        for year_idx, year in enumerate(range(year_begin, year_end)):

            input_name = "%s%d%s.csv" % (input_prefix, year, input_suffix)
            transform_name = 'Transformed_' + input_name

            job = {'cmd': 'transform', 'year': year, 'masknan': masknan,
                   'fillnan': fillnan, 'sites': [],
                   'input_path': os.path.join(input_path, input_name),
                   'output_path': os.path.join(output_path, transform_name)}

            if houston:
                job['sites'] = HOUSTON

            comm.isend(job, dest=n_proc, tag=1)

            n_proc += 1
            n_jobs += 1

            if n_proc == n_procs:
                n_proc = 1

        jobs_done = 0

        while jobs_done < n_jobs:
            req = comm.irecv(tag=2)
            data = req.wait()
            jobs_done += 1

        for nproc in range(1, n_procs):
            req = comm.isend({'cmd': 'shutdown'}, nproc, tag=1)
            req.wait()

        print("Node %d shutting down." % rank)

    else:
        while True:
            req = comm.irecv(source=0, tag=1)
            job = req.wait()

            if job['cmd'] == 'transform':

                print("Got job: %s" % job['year'])
                sys.stdout.flush()

                run_job(job)
                
                print("Finished job: %s" % job['year'])
                sys.stdout.flush()

                result = {'year': job['year']}
                req = comm.isend(result, dest=0, tag=2)
                req.wait()

            elif job['cmd'] == 'shutdown':
                print("Node %d shutting down." % rank)
                sys.stdout.flush()
                return


if __name__ == '__main__':
    plac.call(main)
