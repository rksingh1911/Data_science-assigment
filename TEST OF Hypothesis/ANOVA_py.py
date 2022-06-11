# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 14:23:58 2019

@author: HI
"""

import pandas as pd

df = pd.read_csv("D:/CARRER/My_Course/Data Science Classes/2 Module/7 ANOVA/anova.csv")
df

df.rename(columns={'Data ': 'Data'}, inplace=True)
list(df)

df.shape
df.head()
df.describe() #Describing the dataset

#==============================================================================

from statsmodels.formula.api import ols
import statsmodels.api as sm
lm1 = ols('Data ~ C(Coating)',data=df).fit()
table = sm.stats.anova_lm(lm1, type=1) # Type 1 ANOVA DataFrame

print(table)

#==============================================================================

dfA = df['Data'][df['Coating'] == "A"]
dfB = df['Data'][df['Coating'] == "B"]
dfC = df['Data'][df['Coating'] == "C"]
dfD = df['Data'][df['Coating'] == "D"]

import scipy.stats as stats
stat, p = stats.f_oneway(dfA,dfB,dfC,dfD)

alpha = 0.05

# TOH
# Ho: All means are equal
# H1: Any mean is not equal to other mean

if p < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")

#==============================================================================











""" ANOTHER EXAMPLE FOR REFERENCE
import statsmodels.api as sm
from statsmodels.formula.api import ols

moore = sm.datasets.get_rdataset("Moore", "carData",cache=True) # load data
data = moore.data
data.shape
data = data.rename(columns={"partner.status":"partner_status"}) # make name pythonic

moore_lm = ols('conformity ~ C(fcategory, Sum)*C(partner_status, Sum)',data=data).fit()
table = sm.stats.anova_lm(moore_lm, typ=2) # Type 2 ANOVA DataFrame
print(table)

"""
