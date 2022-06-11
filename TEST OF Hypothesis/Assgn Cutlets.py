# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 22:34:37 2022

@author: Suklesh
"""
#Hypothesis Testing of Cutlets
#To check if there is significant difference between unit A&B or not
#By using Two sample mean test using level of significance(alpha=5%=0.05)

import pandas as pd
data=pd.read_csv("F:\\ExcelR\\Assignments\\Hypothesis testing\\Cutlets.csv")
data.shape
list(data)

'''
#Test of Hypothesis
Ho: UnitA = UnitB ---> No significant difference in diameter of cutlets of two units
H1: UnitA != UnitB ---> Significant difference in diameter of cutlets of two units
'''
uA=data['Unit A']
uB=data['Unit B']
alpha=0.05 #alpha is the level of significance
from scipy.stats import ttest_ind
z,p=ttest_ind(uA,uB)
print(z,p)

if p>alpha:
    print('Accept Ho and Reject H1')
else:
    print('Accept H1 and Reject Ho')
    
#Inference: As we have got to accept Ho that implies there is No significant
#---------- difference between the diameters of Unit A and Unit B