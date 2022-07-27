# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 20:30:42 2022

@author: Hi
"""

# create a line plot
#from matplotlib import pyplot as plt

import pandas as pd
#series = pd.read_csv('daily-minimum-temperatures.csv', header=0, index_col=0,parse_dates=True)
series = pd.read_csv('daily-minimum-temperatures.csv', index_col=0)
series
len(series)


# line plot
series.plot()



#### Histogram and Density Plots
# create a histogram plot
from matplotlib import pyplot
#series = pd.read_csv('daily-minimum-temperatures.csv', header=0, index_col=0)
series.hist()
pyplot.show()

# create a density plot
from pandas import read_csv
from matplotlib import pyplot
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0,parse_dates=True)
series.plot(kind='kde')
pyplot.show()

#### Box and Whisker Plots by Interval
#type(read_csv('daily-minimum-temperatures.csv', header=0, index_col=0,parse_dates=True,squeeze=True))

# create a boxplot of yearly data
from pandas import DataFrame
from pandas import Grouper
from matplotlib import pyplot
series = pd.read_csv('daily-minimum-temperatures.csv', header=0, index_col=0,parse_dates=True,squeeze=True)
groups = series.groupby(Grouper(freq='A'))
years = DataFrame()
for name, group in groups:
    years[name.year] = group.values
years.boxplot()
pyplot.show()

type(years)
list(years)
years[1981]



#### Lag plot

# create a scatter plot
from pandas import read_csv
from matplotlib import pyplot
from pandas.plotting import lag_plot
#series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0,parse_dates=True)
lag_plot(series)
pyplot.show()

# create an autocorrelation plot

from pandas import read_csv
from matplotlib import pyplot
pyplot.figure(figsize = (40,10))
from statsmodels.graphics.tsaplots import plot_acf
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0,parse_dates=True)
plot_acf(series,lags=90)
pyplot.show()


