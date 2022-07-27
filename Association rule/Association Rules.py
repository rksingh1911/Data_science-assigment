# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 20:09:19 2022

@author: Hi
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder
!pip install mlxtend

titanic = pd.read_csv("/content/Titanic.csv")
titanic

df=pd.get_dummies(titanic)
df.head()

#Apriori Algorithm
frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
frequent_itemsets

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.7)
rules
rules.sort_values('lift',ascending = False)

rules.sort_values('lift',ascending = False)[0:20]

rules[rules.lift>1]

rules[['support','confidence']].hist()

rules[['support','confidence','lift']].hist()

import matplotlib.pyplot as plt

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

plt.scatter(rules['support'], rules['confidence'])
plt.show()


import seaborn as sns
sns.scatterplot('support', 'confidence', data=rules, hue='antecedents')

plt.show()

!pip install nsepython


from nsepython import *   
print(indices)

oi_data, ltp, crontime = oi_chain_builder("RELIANCE","latest","full")
print(oi_data)
print(ltp)
print(crontime)


