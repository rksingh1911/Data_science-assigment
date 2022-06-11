# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 23:15:47 2022

@author: Suklesh
"""
#Hypothesis Testing of Avg TAT and recorded TAT (TAT-Turn Around Time)
#To check if there is significant difference between Avg TAT and recorded TAT or not
#By using ANOVA using level of significance(alpha=5%=0.05)

import pandas as pd
data=pd.read_csv("F:\\ExcelR\\Assignments\\Hypothesis testing\\LabTAT.csv")
data.shape
list(data)
'''
#Test of Hypothesis
Ho: l1 = l2 = l3 = l4 ---> All laboratories avg TAT is same
H1: l1 != l2 != l3 != l4 ---> Any one of the laboratories avg TAT among the 4 are not same
'''
l1=data['Laboratory 1']
l2=data['Laboratory 2']
l3=data['Laboratory 3']
l4=data['Laboratory 4']
import scipy.stats as stats
z,p=stats.f_oneway(l1,l2,l3,l4)
print(z,p)
alpha=0.05
if p>alpha:
    print('Accept Ho and Reject H1')
else:
    print('Accept H1 and Reject Ho')
    
#Inference: As per the test of hypothesis we are getting to accept H1,
#---------- so,we can say any one of the laboratories avg TAT among the 4 are not same