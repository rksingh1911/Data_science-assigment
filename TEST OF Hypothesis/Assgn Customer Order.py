# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 23:15:47 2022

@author: Suklesh
"""
#Hypothesis Testing of Tele call data of 4 centers across globe
#To check if there is error in the audit or not
#By using anova using level of significance(alpha=5%=0.05)
import pandas as pd
data=pd.read_csv('F:\\ExcelR\\Assignments\\Hypothesis testing\\Costomer+OrderForm.csv')
data.head()
data.dtypes
list(data)
data.shape
'''
#Test of Hypothesis
Ho: c1 = c2 = c3 = c4 ---> All 4 call centers defective % are varying by 5% from center
H1: c1 != c2 != c3 != c4 ---> Any one of the 4 call centers defective % are NOT varying by 5% from center
'''
#Converting categorical data to numerical by label encoding
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
for i in range(0,4,1):
    data.iloc[:,i]=LE.fit_transform(data.iloc[:,i])
print(data)

c1=data['Phillippines']
c2=data['Indonesia']
c3=data['Malta']
c4=data['India']

import scipy.stats as stats
z,p=stats.f_oneway(c1,c2,c3,c4)
print(z,p)
alpha=0.05
if p>alpha:
    print('Accept Ho and Reject H1')
else:
    print('Accept H1 and Reject Ho')
    
#Inference: As per the test of hypothesis we are getting to accept Ho,
#---------- so,All the 4 call centers defective % are under the 5% level of significance from center