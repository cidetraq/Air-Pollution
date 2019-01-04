import pandas as pd
import numpy as np
from mpi4py import MPI
import plac
import os


def transform(df: pd.DataFrame, year: int, masknan: float = None) -> pd.DataFrame:
    if masknan is None:

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

    if masknan is not None:
        df = df.drop(
            ['co_flag', 'humid', 'humid_flag', 'pm25', 'pm25_flag', 'so2', 'so2_flag', 'solar', 'solar_flag', 'dew',
             'dew_flag', 'redraw', 'co', 'no_flag', 'no2_flag', 'nox_flag', 'o3_flag', 'winddir_flag', 'windspd_flag',
             'temp_flag', 'Longitude', 'Latitude'], axis=1)

        df[df.isna()] = np.nan

    return df


def run_job(job: dict):
    df = pd.read_csv(job['input_path'])
    df = transform(df, job['year'], job['masknan'])
    df.to_csv(job['output_path'])


@plac.annotations(
    input_path=("Path containing the data files to ingest", "option", "P", str),
    input_prefix=("{$prefix}year.csv", "option", "p", str),
    input_suffix=("year{$suffix}.csv", "option", "s", str),
    output_path=("Path to write the resulting numpy sequences / transform cache", "option", "o", str),
    year_begin=("First year to process", "option", "b", int),
    year_end=("Year to stop with", "option", "e", int),
    masknan=("Mask nan rows instead of dropping them", "option", "M", float)
)
def main(input_path: str = '/project/lindner/moving/summer2018/Data_structure_3',
         input_prefix: str = "Data_",
         input_suffix: str = "",
         output_path: str = '/project/lindner/moving/summer2018/2019/data-formatted/parallel',
         year_begin: int = 2000,
         year_end: int = 2018,
         masknan: float = None):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_procs = comm.Get_size()

    if rank == 0:

        n_proc = 0
        n_jobs = 0

        my_jobs = []
        for year_idx, year in enumerate(range(year_begin, year_end)):

            input_name = "%s%d%s.csv" % (input_prefix, year, input_suffix)
            transform_name = 'Transformed_' + input_name

            job = {'cmd': 'transform', 'year': year, 'masknan': masknan,
                   'input_path': os.path.join(input_path, input_name),
                   'output_path': os.path.join(output_path, transform_name)}

            if n_proc != 0:
                comm.isend(job, dest=n_proc, tag=1)
            else:
                my_jobs.append(job)

            n_jobs += 1
            n_proc += 1

            if n_proc == n_procs:
                n_proc = 0

        jobs_done = 0

        for job in my_jobs:
            print("Got job: %s" % job['year'])
            run_job(job)
            print("Finished job: %s" % job['year'])

            jobs_done += 1
            print("Progress: %d/%d" % (jobs_done, n_jobs))

        while jobs_done < n_jobs:

            req = comm.irecv(tag=2)
            data = req.wait()
            jobs_done += 1
            print("Progress: %d/%d" % (jobs_done, n_jobs))

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
                run_job(job)
                print("Finished job: %s" % job['year'])

                result = {'year': job['year']}
                req = comm.isend(result, dest=0, tag=2)
                req.wait()

            elif job['cmd'] == 'shutdown':
                print("Node %d shutting down." % rank)
                return


if __name__ == '__main__':
    plac.call(main)
