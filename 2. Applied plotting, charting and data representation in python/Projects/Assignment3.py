
# coding: utf-8

# # Assignment 3 - Building a Custom Visualization
# 
# ---
# 
# In this assignment you must choose one of the options presented below and submit a visual as well as your source code for peer grading. The details of how you solve the assignment are up to you, although your assignment must use matplotlib so that your peers can evaluate your work. The options differ in challenge level, but there are no grades associated with the challenge level you chose. However, your peers will be asked to ensure you at least met a minimum quality for a given technique in order to pass. Implement the technique fully (or exceed it!) and you should be able to earn full grades for the assignment.
# 
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ferreira, N., Fisher, D., & Konig, A. C. (2014, April). [Sample-oriented task-driven visualizations: allowing users to make better, more confident decisions.](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Ferreira_Fisher_Sample_Oriented_Tasks.pdf) 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems (pp. 571-580). ACM. ([video](https://www.youtube.com/watch?v=BI7GAs-va-Q))
# 
# 
# In this [paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Ferreira_Fisher_Sample_Oriented_Tasks.pdf) the authors describe the challenges users face when trying to make judgements about probabilistic data generated through samples. As an example, they look at a bar chart of four years of data (replicated below in Figure 1). Each year has a y-axis value, which is derived from a sample of a larger dataset. For instance, the first value might be the number votes in a given district or riding for 1992, with the average being around 33,000. On top of this is plotted the 95% confidence interval for the mean (see the boxplot lectures for more information, and the yerr parameter of barcharts).
# 
# <br>
# <img src="readonly/Assignment3Fig1.png" alt="Figure 1" style="width: 400px;"/>
# <h4 style="text-align: center;" markdown="1">  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Figure 1 from (Ferreira et al, 2014).</h4>
# 
# <br>
# 
# A challenge that users face is that, for a given y-axis value (e.g. 42,000), it is difficult to know which x-axis values are most likely to be representative, because the confidence levels overlap and their distributions are different (the lengths of the confidence interval bars are unequal). One of the solutions the authors propose for this problem (Figure 2c) is to allow users to indicate the y-axis value of interest (e.g. 42,000) and then draw a horizontal line and color bars based on this value. So bars might be colored red if they are definitely above this value (given the confidence interval), blue if they are definitely below this value, or white if they contain this value.
# 
# 
# <br>
# <img src="readonly/Assignment3Fig2c.png" alt="Figure 1" style="width: 400px;"/>
# <h4 style="text-align: center;" markdown="1">  Figure 2c from (Ferreira et al. 2014). Note that the colorbar legend at the bottom as well as the arrows are not required in the assignment descriptions below.</h4>
# 
# <br>
# <br>
# 
# **Easiest option:** Implement the bar coloring as described above - a color scale with only three colors, (e.g. blue, white, and red). Assume the user provides the y axis value of interest as a parameter or variable.
# 
# 
# **Harder option:** Implement the bar coloring as described in the paper, where the color of the bar is actually based on the amount of data covered (e.g. a gradient ranging from dark blue for the distribution being certainly below this y-axis, to white if the value is certainly contained, to dark red if the value is certainly not contained as the distribution is above the axis).
# 
# **Even Harder option:** Add interactivity to the above, which allows the user to click on the y axis to set the value of interest. The bar colors should change with respect to what value the user has selected.
# 
# **Hardest option:** Allow the user to interactively set a range of y values they are interested in, and recolor based on this (e.g. a y-axis band, see the paper for more details).
# 
# ---
# 
# *Note: The data given for this assignment is not the same as the data used in the article and as a result the visualizations may look a little different.*

# In[1]:

# Use the following data for this assignment:
import pandas as pd
import numpy as np

np.random.seed(12345)

df = pd.DataFrame([np.random.normal(32000,200000,3650), 
                   np.random.normal(43000,100000,3650), 
                   np.random.normal(43500,140000,3650), 
                   np.random.normal(48000,70000,3650)], 
                  index=[1992,1993,1994,1995])


# In[3]:

import matplotlib 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.cm import ScalarMappable
get_ipython().magic('matplotlib notebook')
############################################# Assignment - Level 'even harder' #############################################
#Determine the color of each bar depending on the value of y regarding the position of the error bar
def bar_color(yv):
    color_data = []
    for i in range(len(df.index)):
        
        if df_mean.iloc[i] - df_yerr.iloc[i] >= yv:
            color_data.append(256) #Red
            
        if df_mean.iloc[i] + df_yerr.iloc[i] <= yv:
            color_data.append(0) #Blue
            
        if (df_mean.iloc[i] - df_yerr.iloc[i] < yv) and (df_mean.iloc[i] + df_yerr.iloc[i] > yv):
            color_rate = abs(yv - (df_mean.iloc[i] - df_yerr.iloc[i]))/(2*df_yerr.iloc[i])
            color_data.append(1-color_rate)
            
    return color_data 

#High of each bar (mean values for each normal distribution) and their corresponding error margins
df_mean = df.apply(np.mean, axis=1)
df_yerr = df.apply(np.std, axis=1)/np.sqrt(df.shape[1])*1.96 #Taken from a forum discussion thread
                                                             #'How to calculate Confidence Interval' - Sophie Greene's answer.

fig, ax = plt.subplots(figsize=(9*0.7,5*0.7))
#Plot a initial horizontal line y=0
yval = 41000
x = [1992-0.8, 1995+0.8]
y = [yval, yval]
plt.plot(x, y, 'k', linewidth=1.2)

#Set the bar chart 
cmap = cm.get_cmap('RdBu_r')
colors = cmap(bar_color(yval))
bars = (plt.bar(df.index.values, df_mean, width = 0.95, yerr=df_yerr, color=colors,
                capsize=5, linewidth=0.5, ecolor='#1C201E', edgecolor='k'))
ax.set_xticks(df.index.values)
ax.tick_params(direction='in', length=2, width=0.8, labelsize=9)
plt.gca().set_title('y = {:.2f}'.format(yval), loc='left', fontsize=9)
plt.axis([1991.3, 1995.7, 0, 52000])

#Add a colorbar
mapp = ScalarMappable(cmap=cmap, norm=plt.Normalize(0,1,max(bar_color(yval))))
mapp.set_array([])
cbar = plt.colorbar(mapp, ticks=[0,0.5,1])
cbar.ax.set_yticklabels(labels=[0,0.5,1])
cbar.ax.tick_params(direction='in', length=3, width=0.8, labelsize=9)

def onclick(event):
    ax.cla()
    plt.cla()
    
    y = [event.ydata, event.ydata]
    plt.plot(x, y, 'k', linewidth=1.2)

    cmap = cm.get_cmap('RdBu_r')
    colors = cmap(bar_color(event.ydata))
    bars = (plt.bar(df.index.values, df_mean, width = 0.95, yerr=df_yerr, color=colors, 
                    capsize=5, linewidth=0.5, ecolor='#1C201E', edgecolor='k'))
    ax.set_xticks(df.index.values)
    ax.tick_params(direction='in', length=2, width=0.8, labelsize=9)
    plt.gca().set_title('y = {:.2f}'.format(event.ydata), loc='left', fontsize=9)
    plt.axis([1991.3, 1995.7, 0, 52000])

    mapp = ScalarMappable(cmap=cmap, norm=plt.Normalize(0,1,max(color_data)))
    mapp.set_array([])
    cbar = plt.colorbar(mapp, ticks=[0,0.5,1])
    cbar.ax.set_yticklabels(labels=[0,0.5,1])
    cbar.ax.tick_params(direction='in', length=3, width=0.8, labelsize=9)

plt.gcf().canvas.mpl_connect('button_press_event', onclick)


# In[ ]:



