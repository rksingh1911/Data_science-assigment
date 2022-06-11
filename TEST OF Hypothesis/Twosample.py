"""
Created on Wed Feb  6 18:42:33 2019

@author: HI
"""
############################################################
import pandas as pd

df1 = pd.read_csv("D:/CARRER/My_Course/Data Science Classes/2 Module/6 Two Sample/cars_100.csv")
df1

df1.shape
df1.head()
df1.describe() #Describing the dataset

#help(scipy.stats)
""" Two Sample Mean Test """
from scipy import stats

# Test of hypothesis
# Ho: mu1 = mu2
# H1: mu1 != mu2

help(stats.ttest_ind)

#stats.ttest_ind(df1['USCARS'],df1['GERMANCARS'])
ztest ,pval = stats.ttest_ind(df1['USCARS'],df1['GERMANCARS']) # By default equal_var = False
    
print("Zcalcualted value is ",ztest.round(4))
print("P-value value is ",pval.round(4))

if pval<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")



"""
ztest ,pval = stats.ttest_ind(df['USCARS'],df['GERMANCARS'],equal_var = False) # By default equal_var = False

"""

############################################################
""" Two Sample Mean Test """

import pandas as pd

df2 = pd.read_csv("D:/CARRER/My_Course/Data Science Classes/2 Module/6 Two Sample/Lungcapdata.csv")
df2
df2.shape


df3 = df2['LungCap'][df2['Gender']=='male']
df4 = df2['LungCap'][df2['Gender']=='female']


from scipy import stats

ztest ,pval = stats.ttest_ind( df3 , df4 ) # By default equal_var = False

print("Zcalcualted value is ",ztest.round(4))
print("P-value value is ",pval.round(4))

if pval<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")



############################################################

from scipy.stats import f
alpha = 0.05 #Or whatever you want your alpha to be.
p_value = scipy.stats.f.cdf(F, df1, df2)
if p_value > alpha:
    # Reject the null hypothesis that Var(X) == Var(Y)
    

#help(stats.ttest_ind)












"""
Two Sample Variance Test i.e  F-Test has to to do

#data generation
# Ex 1: 
import numpy as np
da = np.random.normal(2.3, 0.9, 1000)
da.shape
db = np.random.normal(1.8, 0.7, 1000)
db.shape

from statsmodels.stats.weightstats import ttest_ind
print(ttest_ind(da, db))
"""
