# -*- coding: utf-8 -*-
"""
Created on Tue May 24 11:50:41 2022

@author: Suklesh
"""
#Association rules on books data
import pandas as pd
data=pd.read_csv('F:\\ExcelR\\Assignments\\Association rules\\book.csv')
data.shape
list(data)
data.head()
data.info()
data.values
type(data.values)

book=pd.get_dummies(data)
book.shape    
list(book)

#import mlxtend
from mlxtend.frequent_patterns import apriori,association_rules 
#from mlxtend.preprocessing import TransactionEncoder   

freq_items=apriori(book,min_support=0.1,use_colnames=True)  
freq_items

asr=association_rules(freq_items,metric='lift',min_threshold=0.6) 
asr
asr.sort_values('lift',ascending=False)
asr.sort_values('lift',ascending=False)[0:20]   

asr[asr.lift>1]
asr[['support','confidence','lift']].hist() 

%matplotlib qt
import matplotlib.pyplot as plt
plt.scatter(asr['support'], asr['confidence'])
plt.show()    

import seaborn as sns
sns.scatterplot('support', 'confidence', data=asr, hue='antecedents')
plt.show()
   
    
    
    