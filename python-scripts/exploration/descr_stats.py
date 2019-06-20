#!/usr/bin/env python
# coding: utf-8

# <h2>Preliminary analysis

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
import plac

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
    fiveminutes_daymonths = high_fiveminutes[['day', 'month']]
    high_o3_index = df[['day', 'month']].isin(fiveminutes_daymonths)
    high_o3_days = df.iloc[high_o3_index[high_o3_index['day'] == True].index]
    return high_o3_days


# In[ ]:


def analyze(df_dict, year, output_path):
    year = str(year)
    for df_name in df_dict: 
        df = df_dict[df_name]
        descr_stats = df.describe(include='all')
        #documentation here: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html
        descr_stats.to_csv(output_path+year+'-'+df_name+'-'+'descr_stats.csv')
        hists = df.hist(figsize=(72,72))
        #default is 10 bins for each plot
        plt.savefig(output_path+year+'-'+df_name+'-'+"hist.png")
        plt.clf()
        #To get amount of missing values in each column (do for each df)
        percent_missing = df.isna().mean().round(4) * 100
        percent_missing.to_csv(output_path+year+'-'+df_name+'-'+"percent_missing.csv")
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
            #cleaned[col].plot.hist(density=True, ax=ax)
            col_arr = np.array(cleaned[col].dropna())
            #why is the below code not running on the correct columns... or at all
            if col != 'Latitude' and col != "Longitude":
                (n, bins, patches) = plt.hist(col_arr, range = (0, find_xlim(cleaned[col])))
                print("Number of elements in each bin for "+df_name+"_"+str(col)+": ")
                print(n) 
            plt.xlim(left = 0, right =find_xlim(cleaned[col]))
            plt.savefig(output_path+year+'-'+df_name+'-'+col+"-density.png")
            plt.clf()



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
    testing_path = ("Use this specific file for testing", "option", "T", str)
   
)
def main(testing_path,
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
    
    all_years = []
    #for now just one file, later all years
    if testing_path:
        all_years.append(pd.read_csv(testing_path))
    else:
        for year in range(year_begin,year_end):
            all_years.append(pd.read_csv(input_path+input_prefix+str(year)+".csv"))
    for index, df in enumerate(all_years):
        #Reduce memory usage
        df[['Latitude', 'Longitude', 'co', 'humid', 'no', 'no2', 'nox', 'o3', 'pm25', 'so2', 'solar', 'temp', 'winddir', 'windspd', 'dew']] = df[['Latitude', 'Longitude', 'co', 'humid', 'no', 'no2', 'nox', 'o3', 'pm25', 'so2', 'solar', 'temp', 'winddir', 'windspd', 'dew']].astype('float32')
        #Time series
        df['hour'] = pd.to_datetime(df['epoch'], unit='s').dt.hour
        df['day'] = pd.to_datetime(df['epoch'], unit='s').dt.day
        df['month'] = pd.to_datetime(df['epoch'], unit='s').dt.month
        daytime = df[(df['hour'] > 6) & (df['hour'] < 20)]
        nighttime = df[(df['hour'] < 7) | (df['hour'] > 20)]
        highpol_months = df[(df['month'] >3) & (df['month'] < 11)]
        higho3_days = get_higho3_days(df)
        subsets = {"df":df, "daytime":daytime, "nighttime":nighttime, "highpol_months":highpol_months, "higho3_days":higho3_days}
        year_range = [i for i in range(year_begin, year_end)]
        curr_year = year_range[index]
        analyze(subsets, curr_year, output_path)


# <h2>Site-wise analysis</h2>
# 
# To get site-wise analysis, run the same code on each generated site data file (each will contain all years). 


if __name__ == '__main__':
    plac.call(main)