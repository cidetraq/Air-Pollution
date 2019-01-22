import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from datetime import datetime
import time
import math
import plac
import os
from mpi4py import MPI

n_proc = 0
n_jobs = 0
my_jobs = []
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_procs = comm.Get_size()

def transform(original_data, columns, split):
    newrows=pd.DataFrame()
    for epoch in split:
        subset=original_data[original_data['epoch']==epoch]
        newrow=pd.DataFrame()
        newrow['epoch']=pd.Series(epoch, index=[0])
        newrow['hour']=pd.Series(subset.iloc[0]['hour'], index=[0])
        newrow['day_of_year']=pd.Series(subset.iloc[0]['day_of_year'], index=[0])
        for index, row in subset.iterrows():
            aqs=row['AQS_Code']
            for i, col in enumerate(row):
                if i>0 and i<13:
                    newrow[aqs+"_"+columns[i]]=pd.Series(col, index=[0])     
        #print(newrow)
        newrows=pd.concat([newrows, newrow])
    return newrows            

def transform_container(df, year, columns, epoch_splits, region):
    global n_proc
    global n_jobs
    global n_procs
    for index,split in enumerate(epoch_splits):
        job={'year': year, 'df': df, 'index': index, 'columns': columns, 'region': region, 'split': split}
        if n_proc != 0:
            comm.isend(job, dest=n_proc, tag=1)
        else:
            my_jobs.append(job)
        
        n_jobs += 1
        n_proc += 1

        if n_proc == n_procs:
            n_proc = 0

def run_job(job: dict, output_path):
    newrows = transform(job['df'], job['columns'] , job['split'])
    newrows.to_csv(output_path+'partial/'+'ds2_split_'+str(job['index'])+'_'+job['region']+'_'+str(job['year'])+'.csv', index=False)
    
def make_output(job: dict, output_path, region):
    reshaped_data=pd.DataFrame()
    year=job['year']
    for year_file in job['year_files']:
        df=pd.read_csv(output_path+'/partial/'+year_file)
        reshaped_data=pd.concat([reshaped_data, df])
    reshaped_data.to_csv(output_path+'ds2_'+region+'_'+str(year)+'.csv')

def make_output_container(output_path, year_begin, year_end, region):
    global n_proc
    global n_jobs
    global my_jobs
    global n_procs
    dir_path=output_path+'partial/'
    filenames=os.listdir(dir_path)
    years=list(np.arange(year_begin,year_end))
    for year in years:
        year_files=[filename for filename in filenames if str(year) in filename]
        job={'year_files': year_files, 'year': year}
        if n_proc != 0:
            comm.isend(job, dest=n_proc, tag=1)
        else:
            my_jobs.append(job)
        n_jobs += 1
        n_proc += 1
        if n_proc == n_procs:
             n_proc = 0
        
@plac.annotations(
        input_path=("Path containing the data files to ingest", "option", "P", str),
        input_prefix=("{$prefix}year.csv", "option", "p", str),
        input_suffix=("year{$suffix}.csv", "option", "s", str),
        output_path=("Path to write the resulting numpy sequences / transform cache", "option", "o", str),
        year_begin=("First year to process", "option", "b", int),
        year_end=("Year to stop with", "option", "e", int),
        masknan=("Mask nan rows instead of dropping them", "option", "M", float),
        region=('Region of Texas. Default: houston', 'option', "r", str),
        skip_to_output=('Skip to the concatenation of long rows and output step. Default: False', 'option', 'ff', bool)
)
def main(input_path: str = '/project/lindner/moving/summer2018/2019/data-formatted/mpi-houston/',
         input_prefix: str = "Transformed_Data_",
         input_suffix: str = "",
         output_path: str = '/project/lindner/moving/summer2018/2019/data-formatted/mpi-houston/ds2/',
         year_begin: int = 2000,
         year_end: int = 2001,
         masknan: float = None,
         region: str= "houston",
         skip_to_output: bool=False):
    
    global n_proc
    global n_jobs
    global my_jobs
    global n_procs
    jobs_done=0
    if rank == 0 and skip_to_output==False:
        
        for year_idx, year in enumerate(range(year_begin, year_end)):

            input_name = "%s%d%s.csv" % (input_prefix, year, input_suffix)
            transform_name = 'ds2_'+region+'_'+str(year)
            
            year_params = {'cmd': 'transform', 'year': year, 'masknan': masknan,
                   'input_path': os.path.join(input_path, input_name),
                   'output_path': os.path.join(output_path, transform_name), 'region': region}
            
            df=pd.read_csv(year_params['input_path'])
            epochs=np.array(df['epoch'].unique())
            #Houston splits hard-coded
            epoch_splits=np.split(epochs,105696) 
            columns=df.columns 
            reshaped_data=pd.DataFrame()
            transform_container(df, year, columns, epoch_splits, region)          
            
            
        for job in my_jobs:
            if job['split']%1000==0:
                print("Got job: %s_%s" % (job['index'], job['year']))
            newrows=run_job(job, output_path)
            if job['split']%1000==0:
                print("Finished job: %s_%s" % (job['index'], job['year']))
            
            jobs_done += 1

        while jobs_done < n_jobs:

            req = comm.irecv(tag=2)
            data = req.wait()
            jobs_done += 1

        for nproc in range(1, n_procs):
            req = comm.isend({'cmd': 'shutdown'}, nproc, tag=1)
            req.wait()

        print("Node %d shutting down." % rank)

    elif skip_to_output==False:
        while True:
            req = comm.irecv(source=0, tag=1)
            job = req.wait()

            if job['cmd'] == 'transform':
                if job['split']%1000==0:
                    print("Got job: %s_%s" % (job['split'], job['year']))
                run_job(job)
                if job['split']%1000==0:
                    print("Finished job: %s_%s" % (job['split'], job['year']))

                result = {'year': job['year']}
                req = comm.isend(result, dest=0, tag=2)
                req.wait()

            elif job['cmd'] == 'shutdown':
                print("Node %d shutting down." % rank)
                return
           
    my_jobs=[]
    n_proc = 0
    n_jobs = 0
    make_output_container(output_path, year_begin, year_end, region)
            
    for job in my_jobs:
        print("Got job: %s" % (job['year']))
        make_output(job, output_path, region)
        print("Finished job: %s" % (job['year']))
        jobs_done += 1

    while jobs_done < n_jobs:

        req = comm.irecv(tag=2)
        data = req.wait()
        jobs_done += 1

    for nproc in range(1, n_procs):
        req = comm.isend({'cmd': 'shutdown'}, nproc, tag=1)
        req.wait()

        print("Node %d shutting down." % rank)
        
if __name__ == '__main__':
    plac.call(main)
