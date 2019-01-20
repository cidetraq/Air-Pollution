#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import pickle
import collections
#Euclidean distance of two poin ts a and b. Use lat and long. 
#dist = numpy.linalg.norm(a-b)


# In[16]:


n=4


# In[3]:


#Calculate distances
'''def calc_distances(df):
    distances=[]
    for a in df['AQS_Code'].unique():
        lat=list(df[df['AQS_Code']==a]['Latitude'])[0]
        long=list(df[df['AQS_Code']==a]['Longitude'])[0]
        pointa=[lat,long]
        distances_ab=[]
        for b in df['AQS_Code'].unique():
            if b!=a:
                lat=list(df[df['AQS_Code']==b]['Latitude'])[0]
                long=list(df[df['AQS_Code']==b]['Longitude'])[0]
                pointb=[lat,long]
                dist = np.linalg.norm([pointa,pointb])
                distances_ab.append([dist, b])
        distances.append(distances_ab)
    return distances

'''
# In[4]:


#Local
#df=pd.read_csv('D:/programming-no-gdrive/air-pollution/data-formatted/mpi-houston/Transformed_Data_2000.csv')
#cluster
df=pd.read_csv('/project/lindner/moving/summer2018/2019/data-formatted/mpi-houston/Transformed_Data_2000.csv')


# In[5]:


#distances=calc_distances(df)


# In[6]:


#pickle.dump(distances, open('geo_distances.bin', 'wb'))


# In[7]:


#aqs=df['AQS_Code'].unique()


# In[8]:


'''distances_dict={}
for index, code in enumerate(aqs):
    distances_dict[code]=distances[index]'''


# In[9]:


#pickle.dump(distances_dict, open('geo_distances_dict.bin', 'wb'))


# In[10]:


'''sorted_distances=[]
for site in distances:
    site=sorted(site)
    sorted_distances.append(site)'''


# In[11]:


'''sorted_distances_dict={}
for index, code in enumerate(aqs):
    sorted_distances_dict[code]=sorted_distances[index]
'''

# In[12]:


#pickle.dump(sorted_distances, open('geo_distances_sorted.bin', 'wb'))


# In[13]:


'''def weighting(aqs):
    top_n=sorted_distances_dict[aqs][:n]
    total_distance=sum([site[0] for site in top_n])
    weights=[]
    sites=[]
    for site in top_n:
        weight=site[0]/total_distance
        weight=1-weight
        weights.append([weight])
        sites.append(site[1])
    weights=softmax(weights)
    return (weights, sites)
'''

# In[14]:


'''def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()'''


# In[17]:


'''weights_dict={}
for code in aqs:
    weights=weighting(code)
    weights_dict[code]=weights
weights_dict'''


# In[43]:


#pickle.dump(weights_dict, open('weights_dict.pkl', 'wb'))


# In[45]:


weights_dict=pickle.load(open('../descriptive-output/weights_dict.pkl', 'rb'))


# In[41]:


def get_neighbors(pol,sites,epoch,df):
    pols=[]
    for site in sites:
        site_pol=df[(df['epoch']==epoch) & (df['AQS_Code']==site)]
        site_pol=site_pol[pol]
        if site_pol.isnull().all():
            site_pol=np.nan
            pols.append(site_pol)
        else:
            pols.append(site_pol.values.item())
    return pols


# In[33]:


def count_and_weigh(pols, thresh, weights):
    counts=collections.Counter(pols)
    if counts[np.nan]>=thresh:
        pol_weighted='insuf'
    else:
        pol_weighted=0.0 
        for index, pol in enumerate(pols):
            pol=weights[index]*pol
            pol_weighted+=pol
    return pol_weighted


# In[28]:


def infer_closeby(row, n, thresh):
    #null=row.isnull().any()
    aqs=row['AQS_Code']
    epoch=row['epoch']
    no=np.isnan(row['no'])
    no2=np.isnan(row['no2'])
    nox=np.isnan(row['nox'])
    o3=np.isnan(row['o3'])
    '''if row['no2']==np.nan:
        no2=False
    if row['nox']==np.nan:
        nox=False
    if row['o3']==np.nan:
        o3=False'''
    if no==False or no2==False or nox==False or o3==False:
        profile=weights_dict[aqs]
        sites=profile[1]
        weights=profile[0]
        if no==False:
            no_pols=get_neighbors('no',sites,epoch,df)
            no_weighted=count_and_weigh(no_pols, thresh, weights)
            '''counts=collections.Counter(no_pols)
            if counts[np.nan]>thresh:
                no_pols='insuf'
            else:
                no_pols_weighted=[]
                for index, pol in enumerate(no_pols):
                    pol=weights[index]*pol
                    no_pols_weighted.append(p ol)
                no_pols=no_pols_weighted'''
        else:
            no_weighted=row['no']
        if no2==False:
            no2_pols=get_neighbors('no2', sites,epoch,df)
            no2_weighted=count_and_weigh(no2_pols, thresh, weights)
        else: 
            no2_weighted=row['no2']
        if nox==False:
            nox_pols=get_neighbors('nox', sites,epoch,df)
            nox_weighted=count_and_weigh(nox_pols, thresh, weights)
        else:
            nox_weighted=row['nox']
        if o3==False:
            o3_pols=get_neighbors('o3', sites,epoch,df)
            o3_weighted=count_and_weigh(o3_pols, thresh, weights)
        else:
            o3_weighted=row['o3']
        return [no_weighted,no2_weighted,nox_weighted,o3_weighted]
    else:
        pass


# In[22]:


nulls=df[df.isnull().any(axis=1)]
nulls_no=nulls[nulls['no'].isnull()]


# In[42]:


inferred=nulls_no.apply(lambda x: infer_closeby(x, 4,3), axis=1)
inferred.to_csv('../descriptive-output/inferred_no_hou_2000.csv') 
#pickle.dump(inferred, open('../descriptive-output/inferred_no_hou_2000.pkl', 'rb'))