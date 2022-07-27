# -*- coding: utf-8 -*-
"""
Created on Fri May 27 19:30:22 2022

@author: Suklesh
"""
#Association rules on movies data
import pandas as pd
data=pd.read_csv('F:\\ExcelR\\Assignments\\Association rules\\my_movies.csv')
data.shape
list(data)
data.head()
data.info()
data.values
type(data.values)

movies=pd.get_dummies(data)
movies    
list(movies)

#import mlxtend
from mlxtend.frequent_patterns import apriori,association_rules 
#from mlxtend.preprocessing import TransactionEncoder   

freq_items=apriori(movies,min_support=0.1,use_colnames=True)  
freq_items

asr=association_rules(freq_items,metric='lift',min_threshold=0.6) 
asr
asr.sort_values('lift',ascending=False)
asr.sort_values('lift',ascending=False)[0:20]   

asr[asr.lift>1]
asr[['support','confidence','lift']].hist() 

import matplotlib.pyplot as plt
plt.scatter(asr['support'], asr['confidence'])
plt.show()    

import seaborn as sns
sns.scatterplot('support', 'confidence', data=asr, hue='antecedents')
plt.show()
