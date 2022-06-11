"""
Created on Wed Apr 13 18:11:19 2022
"""

import pandas as pd
df = pd.read_csv("Market.csv")
df.shape
list(df)

df.dtypes
df['Returns']
df['Returns'] = df.Returns.str.replace(',', '').astype(float)


df['Region'].value_counts()

df[['Region','Returns']]

s1 = df['Returns'][df['Region'] == "Africa"]
s2 = df['Returns'][df['Region'] == "Pacific"]
s3 = df['Returns'][df['Region'] == "Canada"]
s4 = df['Returns'][df['Region'] == "Asia"]

s5 = df[['Region','Returns']][(df['Region'] == "Africa") | (df['Region'] == "Pacific") | (df['Region'] == "Canada") | (df['Region'] == "Asia") ]
s5[['Region','Returns']]



import scipy.stats as stats
stat, p = stats.f_oneway(s1,s2,s3,s4)

alpha = 0.05

# TOH
# Ho: All retruns of all regions are same
# H1: Any one region and its return is not same with region


if p < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")



from statsmodels.formula.api import ols
import statsmodels.api as sm
lm1 = ols('Returns ~ C(Region)',data=s5).fit()
table = sm.stats.anova_lm(lm1, type=1) # Type 1 ANOVA DataFrame

print(table)

#-----------------------------------------------------------

