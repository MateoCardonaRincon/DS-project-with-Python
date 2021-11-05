
# coding: utf-8

# # Assignment 2
# 
# Before working on this assignment please read these instructions fully. In the submission area, you will notice that you can click the link to **Preview the Grading** for each step of the assignment. This is the criteria that will be used for peer grading. Please familiarize yourself with the criteria before beginning the assignment.
# 
# An NOAA dataset has been stored in the file `data/C2A2_data/BinnedCsvs_d400/fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89.csv`. This is the dataset to use for this assignment. Note: The data for this assignment comes from a subset of The National Centers for Environmental Information (NCEI) [Daily Global Historical Climatology Network](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/readme.txt) (GHCN-Daily). The GHCN-Daily is comprised of daily climate records from thousands of land surface stations across the globe.
# 
# Each row in the assignment datafile corresponds to a single observation.
# 
# The following variables are provided to you:
# 
# * **id** : station identification code
# * **date** : date in YYYY-MM-DD format (e.g. 2012-01-24 = January 24, 2012)
# * **element** : indicator of element type
#     * TMAX : Maximum temperature (tenths of degrees C)
#     * TMIN : Minimum temperature (tenths of degrees C)
# * **value** : data value for element (tenths of degrees C)
# 
# For this assignment, you must:
# 
# 1. Read the documentation and familiarize yourself with the dataset, then write some python code which returns a line graph of the record high and record low temperatures by day of the year over the period 2005-2014. The area between the record high and record low temperatures for each day should be shaded.
# 2. Overlay a scatter of the 2015 data for any points (highs and lows) for which the ten year record (2005-2014) record high or record low was broken in 2015.
# 3. Watch out for leap days (i.e. February 29th), it is reasonable to remove these points from the dataset for the purpose of this visualization.
# 4. Make the visual nice! Leverage principles from the first module in this course when developing your solution. Consider issues such as legends, labels, and chart junk.
# 
# The data you have been given is near **Ann Arbor, Michigan, United States**, and the stations the data comes from are shown on the map below.

# In[1]:

import matplotlib.pyplot as plt
import mplleaflet
import pandas as pd

def leaflet_plot_stations(binsize, hashid):

    df = pd.read_csv('data/C2A2_data/BinSize_d{}.csv'.format(binsize))

    station_locations_by_hash = df[df['hash'] == hashid]

    lons = station_locations_by_hash['LONGITUDE'].tolist()
    lats = station_locations_by_hash['LATITUDE'].tolist()

    plt.figure(figsize=(8,8))

    plt.scatter(lons, lats, c='r', alpha=0.7, s=200)

    return mplleaflet.display()

leaflet_plot_stations(400,'fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89')


# In[ ]:




# In[4]:

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = (pd.read_csv('data/C2A2_data/BinnedCsvs_d400/fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89.csv')
      .sort_values('Date'))
#Only 2015 data, with temperatures in tenths of celcius
df_2015 = df[df['Date']>='2015']
#Data from 2005 to 2014
df_2005to2014 = (df[(df['Date']!='2008-02-29')
              &(df['Date']!='2012-02-29')
              &(df['Date']<'2015')])

#Minimum and maximum daily records from 2005 to 2014
mint = df_2005to2014.groupby('Date')['Data_Value'].apply(np.min, axis=0)*0.1
maxt = df_2005to2014.groupby('Date')['Data_Value'].apply(np.max, axis=0)*0.1
#Minimum and maximum daily records for 2015
mint_2015 = df_2015.groupby('Date')['Data_Value'].apply(np.min, axis=0)*0.1
maxt_2015 = df_2015.groupby('Date')['Data_Value'].apply(np.max, axis=0)*0.1

#Take the minimum and maximum temperatures for each date in the year through the interval 2005-2014
#i.e. coldest and hottest jan 1st in the interval and so on
mintemp_over_interval = []
maxtemp_over_interval = []
broken_max_temperature = []
broken_min_temperature = []
broken_min_day = []
broken_max_day = []
scatter_maxcolor = []
scatter_mincolor = []

for day in range (0,365):
    
    mintemp = 10000
    maxtemp = -10000
        
    for year_delta in range(0, 10):
        if mint.iloc[day + year_delta*365] < mintemp:
            mintemp = mint.iloc[day + year_delta*365]
        
        if maxt.iloc[day + year_delta*365] > maxtemp:
            maxtemp = maxt.iloc[day + year_delta*365]
        
    mintemp_over_interval.append(mintemp)
    maxtemp_over_interval.append(maxtemp)
    
    #Determine if a temperature record was broken in 2015 regarding the highest or lowest
    #temperature in the same day for the interval 2005-2014
    if mint_2015.iloc[day] < mintemp:
        broken_min_temperature.append(mint_2015.iloc[day])
        broken_min_day.append(day)
        scatter_mincolor.append('#FF6F00')
    if maxt_2015.iloc[day] > maxtemp:
        broken_max_temperature.append(maxt_2015.iloc[day])
        broken_max_day.append(day)
        scatter_maxcolor.append('#000000')
######PLOTING
plt.figure(figsize=(12,6.75), facecolor='#FFFFFF')

plt.plot(maxtemp_over_interval, color='#CA0929', linestyle='-', linewidth=0.9, label='Highest temperatures over 2005-2014 period')
plt.plot(mintemp_over_interval, color='#085E7D', linestyle='-.', linewidth=1.0, label='Lowest temperatures over 2005-2014 period')

plt.gca().fill_between(range(len(mintemp_over_interval)), 
                       mintemp_over_interval, maxtemp_over_interval, 
                       facecolor='#A0A0A0', 
                       alpha=0.25)

plt.scatter(broken_max_day, broken_max_temperature, s=20, c=scatter_maxcolor, label='Broke high temperature records in 2015')
plt.scatter(broken_min_day, broken_min_temperature, s=20, c=scatter_mincolor, label='Broke low temperature records in 2015')

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'Deccember']
pos = np.arange(0, 365, 365/12)
plt.xticks(pos, months, fontsize=12)

plt.xlabel('Date', fontsize=14)
plt.ylabel('Temperature (C°)', fontsize=14)
plt.title('Record temperatures by day through 2005-2015 decade near Ann Arbor, Michigan', fontsize=14)
plt.legend(loc=(0.3,0.02), frameon=False, fontsize=12)

x = plt.gca().xaxis
# rotate the tick labels for the x axis
for item in x.get_ticklabels():
    item.set_rotation(45)
    
#plt.savefig('assignment2.png', bbox_inches='tight')
plt.show()


# In[ ]:



