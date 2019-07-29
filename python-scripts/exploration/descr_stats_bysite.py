#!/usr/bin/env python
# coding: utf-8

# <h2>Preliminary analysis

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from pandas.api.types import is_numeric_dtype
import plac
import sys
import resource


# apply the datetime operations
# 
# Stats: Range, median, average, Q1, Q3, histograms, density plot, percent missing data, total size of dataset
# 
# (For each year 2000-2017)

# In[ ]:


def find_xlim(series):
    if is_numeric_dtype(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        xlim = series.median()+1.5*IQR
        return xlim
    else:
        return max(series)


# In[ ]:


def get_higho3_days(df):
    high_fiveminutes = df[df['o3'] > 40]
    print("Length of high_fiveminutes: "+str(len(high_fiveminutes)) )
    fiveminutes_daymonths = high_fiveminutes[['day', 'month']]
    high_o3_index = df[['day', 'month']].isin(fiveminutes_daymonths)
    #print(high_o3_index[high_o3_index['day'] == True].index)
    #testing
    high_o3_days = df.loc[high_o3_index[high_o3_index['day'] == True].index]
    print(high_o3_days)
    return high_o3_days


# In[ ]:


def analyze(df_dict, year, output_path):
    year = str(year)
    for df_name in df_dict: 
        df = df_dict[df_name]
        descr_stats = df.describe(include='all')
        descr_stats.to_csv(output_path+year+'-'+df_name+'-'+'descr_stats.csv')
        hists = df.hist(figsize=(72,72))
        #default is 10 bins for each plot
        plt.savefig(output_path+year+'-'+df_name+'-'+"hist.png")
        plt.clf()
        #To get amount of missing values in each column (do for each df)
        percent_missing = df.isnull().mean().round(4) * 100
        percent_missing.to_csv(output_path+year+'-'+df_name+'-'+"percent_missing.csv")
        if "overall" in df_name:
            #make plot for monthly missing for current year
            months = df['month'].unique()
            month_nas = pd.DataFrame()
            for month in months:
                nas = df[df['month']==month].isnull().mean().round(4) *100
                month_nas = pd.concat([month_nas,nas], axis=1)
            #plot dataframe of month nulls containing nulls for each column
            month_nas = month_nas.drop(['AQS_Code', 'Latitude', 'Longitude', 'epoch', 'wind_x_dir', 'wind_y_dir', 'hour', 'day_of_year', 'day', 'month', 'year'])
            month_nas = month_nas.transpose()
            fig, ax = plt.subplots()
            month_nas.plot.bar()
            plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
            plt.savefig(output_path+year+'-'+df_name+'-'+"month_missing.png")

                
        #size of dataset
        with open(output_path+df_name+"_nrows", "a") as file:
            file.write(("Number of rows in dataset "+df_name+"-"+year+":"+str(len(df))))
        cleaned = df.dropna(axis = 1, how='all').select_dtypes(['number'])
        for col in cleaned:
            fig, ax = plt.subplots()
            try:
                cleaned[col].dropna().plot.kde(ax=ax, legend=False, title=col, xlim= (0,find_xlim(cleaned[col])))
            except BaseException:
                pass           
            if col != 'Latitude' and col != "Longitude":
                cleaned[col].plot.hist(density=True, ax=ax, range = (0, find_xlim(cleaned[col])) )
            plt.xlim(left = 0, right =find_xlim(cleaned[col]))
            plt.savefig(output_path+year+'-'+df_name+'-'+col+"-density.png")
            plt.clf()
            #
            col_arr = np.array(cleaned[col].dropna())
            if col != 'Latitude' and col != "Longitude":
                (n, bins, patches) = plt.hist(col_arr, range = (0, find_xlim(cleaned[col])))
                print("Number of elements in each bin for "+df_name+"_"+str(col)+": ")
                print(n) 
                sys.stdout.flush()
                plt.clf()
        print("RAM used after "+df_name+" analyze:")
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

# <h2>Main

# In[ ]:


@plac.annotations(
    input_path=("Path containing the data files to ingest", "option", "p", str),
    input_prefix=("{$prefix}year.csv", "option", "P", str),
    input_suffix=("year{$suffix}.csv", "option", "S", str),
    output_path=("Path to write the resulting numpy sequences / transform cache", "option", "o", str),
    year_begin=("First year to process", "option", "b", int),
    year_end=("Year to stop with", "option", "e", int),
    aqsnumerical=("Convert AQS code to numerical", "flag", "A"),
    houston=("Only run for Houston sites", "flag", "H"),
    chunksize=("Process this many records at one time", "option", 'C', int),
    input_file = ("Use this specific file (one site at a time)", "option", "I", str)
   
)
def main(input_file,
         input_path: str = '/project/lindner/air-pollution/level3_data/',
         input_prefix: str = "Data_",
         input_suffix: str = "",
         output_path: str = '/project/lindner/air-pollution/current/2019/descriptive-output/',
         year_begin: int = 2000,
         year_end: int = 2018,
         aqsnumerical: bool = False,
         houston: bool = False,
         chunksize: int = 200000,
        ):
    
    all_files = []
    #main way of using this script is by one site (all years within) at a time
    if input_file:
        #delete junk indexing columns
        data = pd.read_csv(input_file)
        if 'Unnamed: 0' in data.columns:
            data = data.drop(['Unnamed: 0'], axis = 'columns')
        if 'Unnamed: 0.1' in data.columns:
            data = data.drop(['Unnamed: 0.1'], axis = 'columns')
        data.to_csv(input_file)
        all_files.append(data)
    if houston==True:
        #in future add option to run all Houston sites (matching AQS)
        pass
    #legacy (unused) below
    for index, df in enumerate(all_files):
        #Reduce memory usage
        df[['Latitude', 'Longitude', 'co', 'humid', 'no', 'no2', 'nox', 'o3', 'pm25', 'so2', 'solar', 'temp', 'winddir', 'windspd', 'dew']] = df[['Latitude', 'Longitude', 'co', 'humid', 'no', 'no2', 'nox', 'o3', 'pm25', 'so2', 'solar', 'temp', 'winddir', 'windspd', 'dew']].astype('float32')
        #Time series
        df['hour'] = pd.to_datetime(df['epoch'], unit='s').dt.hour
        df['day'] = pd.to_datetime(df['epoch'], unit='s').dt.day
        df['month'] = pd.to_datetime(df['epoch'], unit='s').dt.month
        #add year column which previously was absent
        df['year'] = pd.to_datetime(df['epoch'], unit='s').dt.year
        df_orig = df
        for year in df_orig['year'].unique():
            df = df_orig[df_orig['year'] == year]
            daytime = df[(df['hour'] > 6) & (df['hour'] < 20)]
            print("RAM used after creating daytime df:")
            print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            nighttime = df[(df['hour'] < 7) | (df['hour'] > 20)]
            print("RAM used after creating nighttime df:")
            print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            highpol_months = df[(df['month'] >3) & (df['month'] < 11)]
            print("RAM used after creating highpol_months df:")
            print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            higho3_days = get_higho3_days(df)
            print("RAM used after creating high03_days df:")
            print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            aqs = df['AQS_Code'].iloc[1]
            aqs = str(aqs)
            subsets = {aqs+"_"+"overall":df, aqs+"_"+"daytime":daytime, aqs+"_"+"nighttime":nighttime, aqs+"_"+"highpol_months":highpol_months, aqs+"_"+"higho3_days":higho3_days}
            curr_year = year
            analyze(subsets, curr_year, output_path)


# <h2>Site-wise analysis</h2>
# 
# To get site-wise analysis, run the same code on each generated site data file (each will contain all years). 


if __name__ == '__main__':
    plac.call(main)