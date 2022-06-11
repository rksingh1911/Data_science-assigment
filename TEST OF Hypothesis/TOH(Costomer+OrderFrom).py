"""
Created on Mon May  9 01:17:42 2022

@author: ROHIT SINGH
"""
'''TeleCall uses 4 centers around the globe to process customer order forms. 
They audit a certain %  of the customer order forms. Any error in order form 
renders it defective and has to be reworked before processing.  The manager 
wants to check whether the defective %  varies by centre. Please analyze the 
data at 5% significance level and help the manager draw appropriate inferences'''

#Basically here we have check is there defective % varies by centre or not

#Importing the libraries
import pandas as pd
import numpy as np

# Import .csv file and convert it to a DataFrame object
df = pd.read_csv("E:\\ExcelR\\assigment\\TEST OF Hypothesis\\Costomer+OrderForm.csv")
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

#since here all variable in categorical so we have to go with chi-sqaure test


crosstab = df['Phillippines'].value_counts(),df['Indonesia'].value_counts(),df['Malta'].value_counts(),df['India'].value_counts()

#Hypothesis testing
'''
H0 ==ct1 = ct2 = ct3 = ct4 The defective % does not varies by centre
H1 ==ct1 != ct2 != ct3 != ct4 The defective % does varies by centre
'''
#Chi_square_test
import scipy.stats as stats
result = stats.chi2_contingency(crosstab)
print(result)
#here we got Z value and P value and array
P = 0.28
alpha = 0.05

if P < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")
  
#H0: accecpted hence The defective % does not varies by centre
