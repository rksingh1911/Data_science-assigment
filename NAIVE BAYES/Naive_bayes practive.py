"""
Created on Fri Apr 15 11:54:20 2022

@author: ROHIT SINGH
"""
import pandas as pd
df = pd.read_csv("mushroom.csv")
df.shape
list(df)
df.head()
df.isnull().sum()

#install the researchpy 
#pip install researchpy
import researchpy as rp
pd.crosstab(df['gillattachment'], df['odor'])
table,result = rp.crosstab(df['gillattachment'],df['odor'],test = 'chi-square')
print(table)
print(result)

import scipy.stats as stats
crit = stats.chi2.ppf(q = 0.95, df = 1)
crit.round(3)












