"""
Created on Wed Apr 13 13:19:29 2022

@author: ROHIT SINGH
"""
import pandas as pd
import numpy as np
df = pd.read_csv("breast-cancer-wisconsin-data.csv")
df.shape
df.head()
y = df['diagnosis']
x = df.iloc[:,2:]
from sklearn.preprocessing import StandardScaler
x_scale = StandardScaler().fit_transform(x)
x_scale
type(x_scale)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_scale,y,starify = )







