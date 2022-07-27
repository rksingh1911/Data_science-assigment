"""
Created on Thu Jun 23 19:16:19 2022

@author: ROHIT SINGH
"""

#Business Problem
"Prepare rules for the all the data sets"
'''1) Try different values of support and confidence. Observe the change in number of rules for different support,confidence values
2) Change the minimum length in apriori algorithm
3) Visulize the obtained rules using different plots'''

#Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#loading the data
df = pd.read_csv("E:\\ExcelR\\assigment\\Association rule\\my_movies.csv")
df

#Exploratory Data Analysis
df.head()
df.tail()
df.shape
list(df)
type(df)
df.ndim
df.info()

#describe for all mean mode information
df.describe()

#finding null value
df.isnull()
df.isnull().sum()

#skew for normal distributaion
df.skew()  #so  some  variable skewness are normal and high as well

#corelation
df.corr()

#Data visulazation

#Histogram
df.hist(figsize= (20,7))

#density plot 
df.plot(kind='density', subplots=True, layout=(4,5), figsize=(13,20), sharex=False, sharey=False)
plt.show()

#barplot
df.plot(kind = 'bar',subplots=True, layout=(4,5), figsize=(13,20))

#line plot
df.plot(kind = 'line',figsize = (13,20))

#pair plot
sns.pairplot(df)

#create dummy data in 0 and 1 from
movies=pd.get_dummies(df)
movies    
list(movies)

#frequescy
movies.sum().to_frame('Frequency').sort_values('Frequency',ascending=False)[:25].plot(kind='bar',figsize=(12,8),title="Frequent Items")
plt.show()

#Apriori algorithm
#pip install mlxtend
import mlxtend
from mlxtend.frequent_patterns import apriori,association_rules 
from mlxtend.preprocessing import TransactionEncoder  

#creating empty list
ap_0_5 = {}
ap_1 = {}
ap_5 = {}
ap_1_0 = {}
confidence = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

#creating function
def gen_rules(df,confidence,support):
    ap = {}
    for i in confidence:
        ap_i =apriori(movies,support,True)
        rule= association_rules(ap_i,min_threshold=i)
        ap[i] = len(rule.antecedents)
    return pd.Series(ap).to_frame("Support: %s"%support)

confs = []
for i in [0.001,0.005,0.01,0.05,0.1]:
     ap_i = gen_rules(ap,confidence=confidence,support=i)
     confs.append(ap_i)
     print(confs)
     
#ploting
all_conf = pd.concat(confs,axis=1)
all_conf.plot(figsize=(8,8),grid=True)
plt.ylabel('Rules')
plt.xlabel('Confidence')
plt.show()
"when we have lower confidence then we have higher rules "     
     
freq_items=apriori(movies,min_support=0.1,use_colnames=True)  
freq_items

asr=association_rules(freq_items,metric='lift',min_threshold=0.6) 
asr
asr.sort_values('lift',ascending=False)
asr.sort_values('lift',ascending=False)[0:20]   

asr[asr.lift>1]
asr[['support','confidence','lift']].hist(figsize = (20,8)) 

import matplotlib.pyplot as plt
plt.scatter(asr['support'], asr['confidence'])
plt.show()

import seaborn as sns
sns.scatterplot('support', 'confidence', data=asr, hue='antecedents')
plt.show()

support = asr["support"]
confidence =  asr["confidence"]
lift = asr["lift"]
fig1 = plt.figure()
#%matplotlib qt
ax1 = fig1.add_subplot(111, projection = '3d')
ax1.scatter(support,confidence,lift)
ax1.set_xlabel("support")
ax1.set_ylabel("confidence")
ax1.set_zlabel("lift")






