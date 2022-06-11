# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 23:46:21 2022

@author: Suklesh
"""
#Hypothesis Testing of Buyer ratio between male and female
#To check if there is significant difference between Buyer ratio between male and female or not
#By using Proportioin test using level of significance(alpha=5%=0.05)

import pandas as pd
data=pd.read_csv("F:\\ExcelR\\Assignments\\Hypothesis testing\\BuyerRatio.csv")
data.shape
data.head
list(data)

'''
#Test of Hypothesis
Ho: Mu = Fu ---> Male and Female people buyer ratio is similar
H1: Mu != Fu ---> Male and Female people buyer ratio is NOT similar
'''
x1=50+142+131+70
x2=435+1523+1356+750

#East Region
p1e=50/x1
p2e=435/x2

print(p1e,p2e)

import numpy as np
count=np.array([p1e,p2e])
numb=np.array([x1,x2])

from statsmodels.stats.proportion import proportions_ztest
stat,pe=proportions_ztest(count,numb)
print(pe)

alpha =0.05
'''
Ho: Male and Female buying ratio is same
H1: Male and Female buying ratio are NOT same
'''

if pe>alpha:
    print('Accept Ho and Reject H1')
else:
    print('Accept H1 and Reject H0')

#West Region
p1w=142/x1
p2w=1523/x2

print(p1w,p2w)

import numpy as np
count=np.array([p1w,p2w])
numb=np.array([x1,x2])

from statsmodels.stats.proportion import proportions_ztest
stat,pw=proportions_ztest(count,numb)
print(pw)

alpha =0.05
'''
Ho: Male and Female buying ratio is same
H1: Male and Female buying ratio are NOT same
'''

if pw>alpha:
    print('Accept Ho and Reject H1')
else:
    print('Accept H1 and Reject H0')

#North Region
p1n=131/x1
p2n=1356/x2

print(p1n,p2n)

import numpy as np
count=np.array([p1n,p2n])
numb=np.array([x1,x2])

from statsmodels.stats.proportion import proportions_ztest
stat,pn=proportions_ztest(count,numb)
print(pn)

alpha =0.05
'''
Ho: Male and Female buying ratio is same
H1: Male and Female buying ratio is NOT same
'''

if pn>alpha:
    print('Accept Ho and Reject H1')
else:
    print('Accept H1 and Reject H0')

#South Region
p1s=70/x1
p2s=750/x2

print(p1s,p2s)

import numpy as np
count=np.array([p1s,p2s])
numb=np.array([x1,x2])

from statsmodels.stats.proportion import proportions_ztest
stat,ps=proportions_ztest(count,numb)
print(pw)

alpha =0.05
'''
Ho: Male and Female buying ratio is same
H1: Male and Female buying ratio are NOT same
'''

if ps>alpha:
    print('Accept Ho and Reject H1')
else:
    print('Accept H1 and Reject H0')