
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# # Assignment 4 - Hypothesis Testing
# This assignment requires more individual learning than previous assignments - you are encouraged to check out the [pandas documentation](http://pandas.pydata.org/pandas-docs/stable/) to find functions or methods you might not have used yet, or ask questions on [Stack Overflow](http://stackoverflow.com/) and tag them as pandas and python related. And of course, the discussion forums are open for interaction with your peers and the course staff.
# 
# Definitions:
# * A _quarter_ is a specific three month period, Q1 is January through March, Q2 is April through June, Q3 is July through September, Q4 is October through December.
# * A _recession_ is defined as starting with two consecutive quarters of GDP decline, and ending with two consecutive quarters of GDP growth.
# * A _recession bottom_ is the quarter within a recession which had the lowest GDP.
# * A _university town_ is a city which has a high percentage of university students compared to the total population of the city.
# 
# **Hypothesis**: University towns have their mean housing prices less effected by recessions. Run a t-test to compare the ratio of the mean price of houses in university towns the quarter before the recession starts compared to the recession bottom. (`price_ratio=quarter_before_recession/recession_bottom`)
# 
# The following data files are available for this assignment:
# * From the [Zillow research data site](http://www.zillow.com/research/data/) there is housing data for the United States. In particular the datafile for [all homes at a city level](http://files.zillowstatic.com/research/public/City/City_Zhvi_AllHomes.csv), ```City_Zhvi_AllHomes.csv```, has median home sale prices at a fine grained level.
# * From the Wikipedia page on college towns is a list of [university towns in the United States](https://en.wikipedia.org/wiki/List_of_college_towns#College_towns_in_the_United_States) which has been copy and pasted into the file ```university_towns.txt```.
# * From Bureau of Economic Analysis, US Department of Commerce, the [GDP over time](http://www.bea.gov/national/index.htm#gdp) of the United States in current dollars (use the chained value in 2009 dollars), in quarterly intervals, in the file ```gdplev.xls```. For this assignment, only look at GDP data from the first quarter of 2000 onward.
# 
# Each function in this assignment below is worth 10%, with the exception of ```run_ttest()```, which is worth 50%.

# In[6]:


import pandas as pd
import numpy as np
from scipy.stats import ttest_ind


# In[7]:


# Use this dictionary to map state names to two letter acronyms
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}
universities_regions_df = pd.read_table('university_towns.txt', header=None).rename(columns={0: 'Data column'})

def get_list_of_university_towns(ur_df=universities_regions_df):
    cols = []
    for line in ur_df['Data column']:
        if '[e' in line:
            st = line.split('[e')[0]
        else:
            reg_name = line.split('(')[0].strip()
            cols.append([st, reg_name])
            
    return pd.DataFrame(cols, columns=['State', 'RegionName'])
get_list_of_university_towns()


# In[99]:


GDP = pd.read_excel('gdplev.xls', header=None, skiprows=list(np.arange(0, 8)), usecols=[4, 5, 6])
def get_recession_start(GDP=GDP):    
    col_names={4: 'Quarter', 5: 'GDP16', 6: 'GDP09'}
    GDP = (GDP.rename(columns=col_names)
           .set_index('Quarter')['2000q1':])
    
    for i in range(len(GDP) - 2):
        if GDP['GDP16'].iloc[i+2] < GDP['GDP16'].iloc[i+1] < GDP['GDP16'].iloc[i]:
            return GDP.iloc[i].name
get_recession_start()


# In[115]:


def get_recession_end(GDP=GDP):
    col_names={4: 'Quarter', 5: 'GDP16', 6: 'GDP09'}
    GDP = (GDP.rename(columns=col_names)
           .set_index('Quarter')['2000q1':])

    recession_begins = False
    for i in range(len(GDP) - 2):
        if GDP['GDP16'].iloc[i+2] < GDP['GDP16'].iloc[i+1] < GDP['GDP16'].iloc[i]:
            recession_begins = True
        if recession_begins == True:
            if GDP['GDP16'].iloc[i+2] > GDP['GDP16'].iloc[i+1] > GDP['GDP16'].iloc[i]:
                return GDP.iloc[i+2].name
get_recession_end()


# In[158]:


def get_recession_bottom(GDP=GDP):
    col_names={4: 'Quarter', 5: 'GDP16', 6: 'GDP09'}
    GDP = (GDP.rename(columns=col_names)
           .set_index('Quarter')['2000q1':])
    
    recession_begins = False
    rec_bottom = []
    for i in range(len(GDP) - 2):
        if GDP['GDP16'].iloc[i+2] < GDP['GDP16'].iloc[i+1] < GDP['GDP16'].iloc[i]:
            recession_begins = True
            
        if recession_begins == True:
            if not (GDP['GDP16'].iloc[i+2] > GDP['GDP16'].iloc[i+1] > GDP['GDP16'].iloc[i]):
                if GDP['GDP16'].iloc[i+1] < GDP['GDP16'].iloc[i]:
                    rec_bottom.append([GDP['GDP16'].iloc[i+1], GDP.iloc[i+1].name])

    return min(rec_bottom)[1]
get_recession_bottom()


# In[159]:


housing_df = pd.read_csv('City_Zhvi_AllHomes.csv')
def convert_housing_data_to_quarters(housing_df=housing_df):
    housing_df = (housing_df.replace(states)
                  .set_index(['State', 'RegionName'])
                  .loc[:,'2000-01':])
    
    for year_g, year_f in housing_df.groupby(lambda x: str(x[0:4]), axis=1):
        for i in range(4):
            housing_df[year_g+'q'+str(i+1)] = np.nanmean(year_f.iloc[:,3*i:3*(i+1)], axis=1)
        
    return housing_df.loc[:,'2000q1':'2016q3']
convert_housing_data_to_quarters()


# In[160]:


def run_ttest():
    housing_df = convert_housing_data_to_quarters()
    u_towns = get_list_of_university_towns().set_index(['State', 'RegionName'])
    
    housing_price_change = abs(housing_df[get_recession_bottom()].sub(housing_df[get_recession_start()]))
    utowns_housing_price_change = housing_price_change.loc[u_towns.index]
    non_utowns_housing_price_change = housing_price_change.drop(labels=u_towns.index)
    
    t_test = ttest_ind(utowns_housing_price_change, non_utowns_housing_price_change, nan_policy='omit')
    pvalue = t_test[1]
    difference = True if pvalue < 0.01 else False
    better = 'university town' if t_test[0] < 0 else 'non-university town'
    
    return (difference, pvalue, better)
run_ttest()


# In[ ]:




