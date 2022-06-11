"""
Created on Tue Apr 26 01:01:49 2022

@author: ROHIT SINGH
"""

import numpy as np
import pandas as pd
#loading the data

df = pd.read_csv("E:\\ExcelR\\assigment\\MLR\\50_Startups.csv")
df.shape
type(df)
list(df)
df.ndim
df

x = df.iloc[:,1:5]

y = df.iloc['Profit']













