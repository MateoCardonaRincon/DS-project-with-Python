
# coding: utf-8

# # Foreign investment in Colombian coffee during the last two decades
# 
# #### 1. Region and domain
# Colombia - Coffee trade and economy
# 
# #### 2. Research Question
# How has the production and export of Colombian coffee been in the last two decades and how have the main foreign countries invested in this product?
# 
# #### 3. Links
# There are two downloadable datasets in .xlsx format at *Federación Nacional de Cafeteros de Colombia* official website https://federaciondecafeteros.org/wp/estadisticas-cafeteras/, one on the left down the page for
# *Prices, area, and production of coffee* (**Precios, área y producción del café** in spanish)
# and the other, next to the right, for *Colombian coffee exports* (**Exportaciones de café colombiano** in spanish).
# 
# On the International Coffee Organization webpage, http://www.ico.org/new_historical.asp?section=StatisticsOther, can be downloaded different datasets about coffee import/export data. I took *ICO Composite & Group
# Indicator Prices - Monthly Averages* dataset, which contains the historical prices of coffee in the international market. This dataset is also in the *Prices, area, and production of coffee* dataset of the *Federación Nacional de Cafeteros de Colombia*, sheet '6. Precio OIC Mensual'.
# 
# #### 4. Image
# <img src="files/Foreign_investment_in_Colombian_coffee.png" width="750">
# Run the code cells bellow - Please run this code in the Coursera - Jupyter Notebook to avoid version issues.
# 
# #### 5. Justification - Discussion (1-2 paragraphs)
# This visualization aims to explain some features of the investment made by the main Colombian coffee importing countries. To do this, I searched the official websites of the *Federación Nacional de Cafeteros de Colombia* and the *International Coffee Organization*, finding several related tables within their .xlsx files. The datasets that I picked up relate mainly to prices, quantities, and countries of destination of Colombian coffee exports.
# 
# The upper panel of the figure shows the average annual investment, in billions of US dollars, that the most importing countries of Colombian coffee have made during the last two decades. It can be observed that the United States has the highest investment with an average of around 800 billion dollars annually, followed by countries like Japan, Germany, Canada, and Belgium, that reach prices of investment over the 100 USD billions per year.
# The bottom panel at the left y-axis (purple solid line) shows the volume of the exportations of Colombian coffee in thousands of 60 Kg bags. This line graph exhibits a tendency to present a maximum peak at the last 2 or 3 months of every year, and it shows a decrease in the period 2009-2012, with a recovery at 2010 ends. The right y-axis (yellow dashed line) depicts the prices of these exports in millions of US dollars, showing an increasing tendency reaching a maximum peak at the end of 2010 and 2011.
# It is to notice that the export prices annual pikes show a correlation with the pikes of volume exports, which is a result that can be expected. However, any kind of correlation relation has not been calculated.

# In[133]:

fig


# In[128]:

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
get_ipython().magic('matplotlib notebook')

col_names = {'exports':[None] + list(map(str, np.arange(2000, 2020))), 
             'ico':['Year', 'ICO composite indicator', 'Colombian Milds',
                    'Other Milds', 'Brazilian Naturals']}


# In[129]:

#Colombian coffee export volumes in thousands of 60 Kg bags
#Download dataset at:
# https://federaciondecafeteros.org/wp/estadisticas-cafeteras/
# -> 'Exportaciones de café colombiano' dataset - Sheet: '1. Total_Volumen'
volume_df = (pd.read_excel('Exports.xlsx', '1. Total_Volumen', skiprows=np.concatenate([np.arange(8,512), np.arange(752,767)]),
                           header=7)
             .set_index('MES') #Monthly data
             )

#Colombian coffee export values in millions of US dollars
# -> 'Exportaciones de café colombiano' dataset - Sheet: '2. Total_Valor'
values_df = (pd.read_excel('Exports.xlsx', '2. Total_Valor', skiprows=np.concatenate([np.arange(8,512), np.arange(752,767)]),
                           usecols=[0,1], header=7)
             .set_index('Mes') #Monthly data
             )


# In[130]:

#Colombian coffee export voluemes in thousands of 60 Kg bags for different countries through last two decades
# -> 'Exportaciones de café colombiano' dataset - Sheet: '6. Destino_Vol'
exports_df = (pd.read_excel('Exports.xlsx', '6. Destino_Vol', skiprows= [7, 8, 12, 29, 34], usecols=np.arange(0, 22),
                            index_col=0, header=6)
              .set_index('PAISES')
              .T
              .set_index(np.arange(2000, 2020))
          )

exports_df = exports_df*1000 #Convert the values from 'thousands of bags' to 'units of 60 Kg'
curr_cols = exports_df.columns.values
exports_df.rename(columns={curr_cols[0]:'United States', curr_cols[1]:'Canada',
                           curr_cols[2]:'Argentina', curr_cols[3]:'Germany',
                           curr_cols[4]:'Belgium', curr_cols[5]:'Italy',
                           curr_cols[6]:'United Kingdom', curr_cols[7]:'Sweden',
                           curr_cols[8]:'Netherlands', curr_cols[9]:'Spain',
                           curr_cols[10]:'Finland', curr_cols[11]:'France',
                           curr_cols[12]:'Denmark', curr_cols[13]:'Poland',
                           curr_cols[14]:'Portugal', curr_cols[15]:'Austria',
                           curr_cols[16]:'Greece', curr_cols[17]:'Norway',
                           curr_cols[18]:'Switzerland', curr_cols[19]:'Japan',
                           curr_cols[20]:'South Korea', curr_cols[21]:'Australia',
                           curr_cols[22]:'Other countries'
                          }, inplace=True)
exports_df.columns.name = ''


# In[131]:

#International Coffee Organization indicator prices in USD cents per pound (0.453592 Kg)
#Download dataset at:
# https://federaciondecafeteros.org/wp/estadisticas-cafeteras/
# -> 'Precios, área y producción del café' dataset - Sheet '6. Precio OIC Mensual'
#or at:
# http://www.ico.org/new_historical.asp?section=StatisticsOther 
# -> ICO Composite & Group Indicator Prices - Monthly Averages dataset

ico_df = (pd.read_excel('coffee_prices.xlsx', '6. Precio OIC Mensual',
                                skiprows=np.concatenate([np.arange(0, 6), np.arange(247,259)]), usecols=[1, 2, 5, 8, 11], 
                                names=col_names['ico']
                       )
          .set_index('Year')
          .groupby(pd.Grouper(freq='A'))
          .apply(np.mean) #Get the mean of the monthly prices for each year
          .set_index(np.arange(2000, 2020))
                 )

#This dataframe represents the annual average values of ICO indicators for the last two decades
ico_df.index.name = 'Year'
bagprice_df = ico_df*(60/0.4536)/100  #Price of a 60 Kg bag of coffe in USD (not cents of USD)
                                                #according to ICO indicators

#Number of 60 Kg bags of coffee (for each country) in the dataframe "exports_df"
#multiplied by the export prices, according to ICO indicators, for each year of the last two decades
expval_df = exports_df.multiply(bagprice_df['Colombian Milds'], axis=0) # #bags_each_country*ICOprice per year
                                                                        # i.e. foreign investment per country and year

#Mean annual investment per country through the last two decades 
#Mean over all 20 year per country
mean_df = expval_df.apply(np.mean, axis=0)
mean_df.sort_index(inplace=True)


# In[132]:

#Plotting the data
plt.style.use('seaborn-dark') #Define a plot style

fig = plt.figure(figsize=(10, 5))
gs = gridspec.GridSpec(2,1)

#Top panel - barchart of foreign investment in Colombian coffee
ftop = plt.subplot(gs[0, 0]) #Define top panel

bars = ftop.bar(np.arange(len(mean_df.index)), mean_df/(10**6), color='#138D75', width=0.9) #
ftop.tick_params(length=0, labelsize=12)
ftop.set_title('Foreign investment in Colombian coffee during the last two decades', loc='center', fontsize=14)
plt.axis([-0.7, 22.7, 0, 1000])
plt.ylabel('Billions of USD *', fontsize=12)
plt.xticks([],[])
ftop.set_yscale('symlog') #symlog: symetric logaritmic scale
ftop.text(-0.5, 500, '* Average annual investment', color='#17202A', fontsize=10)

dec = 1
for i in range(4):
    ftop.axhline(dec, linestyle='--', color='k', linewidth=0.5, alpha=0.25)
    dec *= 10
    
countries = mean_df.index.values
for i, rect in enumerate(bars):
        h = rect.get_width()
        nchars = len(countries[i])
        ftop.text(rect.get_x() + 0.55, 0.1,
                countries[i].upper(), verticalalignment='bottom', fontsize=10, color='#17202A', weight='bold', rotation=90)

#Define bottom panel - left labeled plot
fbotL = plt.subplot(gs[1, 0])

fbotL.plot(volume_df, '-', color='#633974', linewidth=1.4, label='Export volume')
plt.xlabel('Year', color='k', fontsize=12)
plt.ylabel('Thousands of 60 Kg bags', color='k', fontsize=12)
#Ticks
date_ticks = np.arange('2000-01-01', '2021-01-01', dtype='datetime64[Y]')
date_ticklabels = ['2000', '', '2002', '', '2004', '', '2006', '', '2008', '', '2010', '', '2012', '', '2014',
                        '', '2016', '', '2018', '', '2020']
plt.yticks([0, 750, 1500], ['0', '750', '1500'])
plt.xticks(date_ticks, date_ticklabels)

fbotL.tick_params(axis='x', length=2, direction='in', labelsize=12)
fbotL.tick_params(axis='y', length=0, direction='in', labelsize=12)
fbotL.legend(loc=(0.02, 0.84), fontsize=12, frameon=False)

#Define bottom panel - left labeled plot - clone axes with twinx
fbotR = fbotL.twinx()

fbotR.plot(values_df, '--', color='#F7B512', linewidth=1.3, label='Export prices')
plt.xlabel('Year', color='k', fontsize=12)
plt.ylabel('Millions of USD', color='k', fontsize=12)
#Ticks
plt.yticks([0, 100, 200, 300, 400], ['0', '100', '200', '300', '400'])
plt.xticks(date_ticks, date_ticklabels)
fbotR.tick_params(axis='x', length=2, direction='in', labelsize=12)
fbotR.tick_params(axis='y', length=0, direction='in', labelsize=12)
fbotR.grid(axis='y', linestyle='--', color='k', linewidth=0.4, alpha=0.25)
fbotR.legend(loc=(0.79, 0.03), fontsize=12, frameon=False)
plt.tight_layout()

plt.subplots_adjust(wspace=None, hspace=0.09)
plt.savefig('Foreign_investment_in_Colombian_coffee.png')

