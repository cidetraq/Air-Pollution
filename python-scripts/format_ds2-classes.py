import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from datetime import datetime
import time
import math
import plac
import os
from mpi4py import MPI

class ds2_scheduler():
    def __init__(self, comm, n_procs, input_path: str,
         input_prefix: str,
         output_path: str,
         year_begin: int,
         year_end: int,
         region: str):
        self.comm=comm
        self.n_procs=n_procs
        self.input_path=input_path
        self.input_prefix=input_prefix
        self.years=range(year_begin,year_end)
        self.region=region
        self.output_path=output_path
        self.jobs_queued=0
        self.jobs_done=0
        
    def _schedule_container(self):
        for year in self.years:
            input_name = "%s%s.csv" % (self.input_prefix, year)
            #output_name = 'ds2_'+region+'_'+str(year)
            self._transform_year(year, input_name, self.input_path,self.output_path, self.region)
            
    def _transform_year(self, year, input_name, input_path, output_path, region):
        df=pd.read_csv(input_path+input_name)
        epochs=np.array(df['epoch'].unique())
        #split_length=len(epochs)//(n_procs-1)
        split_divisions=self.n_procs-1
        epoch_splits=np.array_split(epochs, split_divisions)
        for index,split in enumerate(epoch_splits):
            job={'year': year, 'input_name': input_name, 'split_index': index, 'split_divisions': split_divisions}
            self.comm.isend(job, dest=index, tag=1)
            self.jobs_queued+=1
            
    def make_output_container(self):
        dir_path=self.output_path+'partial/'
        filenames=os.listdir(dir_path)
        years=list(np.arange(year_begin,year_end))
        for index,year in enumerate(years):
            year_filenames=[filename for filename in filenames if str(year) in filename]
            job={'year': year, 'filenames': year_filenames}
            self.comm.isend(job, dest=index+1)
            self.jobs_queued+=1

class ds2_worker():
    def __init__(self, split_divisions: int, split_index: int, input_name: str, input_path: str,
         output_path: str,
         year: int,
         region: str):
        self.split_divisions=split_divisions
        self.split_index=split_index
        self.input_path=input_path
        self.input_name=input_name
        self.year=year
        self.region=region
        self.output_path=output_path
        self.split, self.df=self._load_df()
        self.columns=self.df.columns
        
    def _load_df(self):
        print('INPUT NAME:')
        print(self.input_name)
        df=pd.read_csv(self.input_path+self.input_name)
        epochs=np.array(df['epoch'].unique())
        epoch_splits=np.array_split(epochs, self.split_divisions)
        split=epoch_splits[self.split_index]
        combined=pd.DataFrame()
        for epoch in split:
            sub=df[df['epoch']==epoch]
            combined=pd.concat([combined, sub])
        return split, combined
    
    def _transform(self):
        newrows=pd.DataFrame()
        for epoch in self.split:
            subset=self.df[self.df['epoch']==epoch]
            newrow=pd.DataFrame()
            newrow['epoch']=pd.Series(epoch, index=[0])
            newrow['hour']=pd.Series(subset.iloc[0]['hour'], index=[0])
            newrow['day_of_year']=pd.Series(subset.iloc[0]['day_of_year'], index=[0])
            for index, row in subset.iterrows():
                aqs=row['AQS_Code']
                for i, col in enumerate(row):
                    if i>0 and i<13:
                        newrow[aqs+"_"+self.columns[i]]=pd.Series(col, index=[0])     
            #print(newrow)
            newrows=pd.concat([newrows, newrow])
        return newrows   
    
    def _save(self, newrows): newrows.to_csv(self.output_path+'partial/'+'ds2_split_'+str(self.split_index)+'_'+self.region+'_'+str(self.year)+'.csv', index=False)
        
    def _run(self):
        print("Got job: %s_%s" % (self.split_index, self.year))
        newrows=self._transform()
        self._save(newrows)
        print("Finished job: %s_%s" % (self.split_index, self.year))
        
    def make_output(self, job):
        reshaped_data=pd.DataFrame()
        year=job['year']
        for year_file in job['filenames']:
            df=pd.read_csv(output_path+'/partial/'+year_file)
            reshaped_data=pd.concat([reshaped_data, df])
        reshaped_data.to_csv(output_path+'ds2_'+region+'_'+str(year)+'.csv')
        
        
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

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_procs = comm.Get_size()
    print('RANK:        '+str(rank))
    print(year_begin)
    print(year_end)
    if rank == 0 and skip_to_output==False:
        scheduler=ds2_scheduler(comm, n_procs, input_path=input_path,
         input_prefix=input_prefix,
         output_path=output_path,
         year_begin=year_begin,
         year_end=year_end,
         region=region)
        scheduler._schedule_container() 
        while scheduler.jobs_done<(n_procs-1):
            #Update jobs_done counter, no other functionality
            req = comm.irecv()
            data = req.wait()
            scheduler.jobs_done+=1
        
    if rank > 0 and skip_to_output==False:
        req=comm.irecv()
        job=req.wait()
        if job['split_divisions']:
            worker=ds2_worker(split_divisions=job['split_divisions'], split_index=job['split_index'], input_name=job['input_name'], input_path=input_path,
             output_path=output_path,
             year=job['year'],
             region=region)
            worker._run()
            #Send information to scheduler to update jobs_done
            req = comm.isend({'job': True, 'rank': rank}, dest=0)
            req.wait()
        else:
            output_step=True
    
    if rank == 0:
        scheduler.make_output_container()
        
    if rank > 0 or output_step==True:
        req=comm.irecv()
        job=req.wait()
        print("Got job: output_%s" % (job['year']))
        worker.make_output(job)
        print("Finished job: output_%s, shutting down..." % (job['year']))
        result={'job': True}
        req = comm.isend(result, dest=0)
        return


        
if __name__ == '__main__':
    plac.call(main)
