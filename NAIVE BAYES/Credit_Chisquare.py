# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 21:28:20 2019

@author: HI
"""

import pandas as pd
df = pd.read_csv("D:\\CARRER\\My_Course\\Data Science Classes\\2 Module\\8 Test of Independence\\credit_1.csv")
list(df)

df.shape

# install first in anaconda - pip install researchpy

pd.crosstab(df['Cards'], df['Ethnicity'])

#pip install researchpy
import researchpy as rp
table, results = rp.crosstab(df['Cards'], df['Ethnicity'], test= 'chi-square')
print(table)
print(results)

# Chi square table values for given alpha and degrees of freedom
import scipy.stats as stats
crit = stats.chi2.ppf(q = 0.95, df = 1)
crit.round(4)


#=============================================

import pandas as pd
df = pd.read_csv("Creditcard.csv")
df

#pip install researchpy
import researchpy as rp

table, results = rp.crosstab(df['Gender'], df['Married'], test= 'chi-square')

print(table)
print(results)

alpha = 0.05

# TOH
# Ho: There is no relationship
# H1: There is some relationship

p=0.8033
if p < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")
#-------------------------------------------------

import scipy.stats as stats
crosstab = pd.crosstab(df['Gender'], df['Married'])
crosstab
stats.chi2_contingency(crosstab)

p = 0.8836532332904176
if p < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")


