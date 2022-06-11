"""
Created on Sun Mar  6 20:49:51 2022

"""
#Anova
# Load libraries
from sklearn import datasets
# Load digits dataset
iris = datasets.load_iris()

iris.data
iris.target
iris.target_names
iris.feature_names



import pandas as pd
df = pd.DataFrame(iris.data)
df.head()
df.describe()
df.shape
list(df)

df[0].hist()
df[1].hist()
df[2].hist()
df[3].hist()

df.iloc[3:5,0:3]
df[0]

#------------------------------------------------------------------------------
############ one way ANOVA #####################
import scipy.stats as stats
stat, p = stats.f_oneway(df.iloc[:,0], df.iloc[:,1],df.iloc[:,2],df.iloc[:,3])


alpha = 0.05

# TOH
# Ho: All means are equal
# H1: Any mean is not equal to other mean

if p < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")

#------------------------------------------------------------------------------
