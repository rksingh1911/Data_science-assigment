"""
Created on Mon May  9 07 2022

@author: ROHIT SINGH
"""
#Problem Statment
'''Sales of products in four different regions is tabulated for males and females. 
Find if male-female buyer rations are similar across regions'''

#Basically here we have check Is the male-female buyer rations are similar across region or not 

#Importing the libraries
import pandas as pd
import numpy as np

# Import .csv file and convert it to a DataFrame object
df = pd.read_csv("E:\\ExcelR\\assigment\\TEST OF Hypothesis\\BuyerRatio.csv")
df.head
df.shape
type(df)
list(df)
df.ndim
df.info()

# finding missing values
df.isnull().sum()

#Exploratory Data Analysis
df.describe()

#Hypothesis testing
'''
H0 == The male-female buyer rations are similar across regions
H1 == The male-female buyer rations are not similar across regions
'''
# as we know here we have only two rom with 5 colum or continous and categorical data
# so we can go with propotion_test , anova_test, or chi_square_test
#we will go with Anova test

#Anova-test
import scipy.stats as stats
stat, p = stats.f_oneway(df['East'],df['West'],df['North'],df['South'])
print(stat)
print(p)

alpha = 0.05 # 5% level of significance

if p < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")

#H0 is Accepted hence male and female buyer ratio are same
