# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:17:46 2022

@author: Suklesh
"""
#Decision Tree
import pandas as pd
import numpy as np
import seaborn as sns
data=pd.read_csv("F:\\ExcelR\\Assignments\\Decision Trees\\Company_Data.csv")
data.head()
list(data)
data.shape
data.dtypes
X1=data['ShelveLoc']
data.drop(['ShelveLoc'],axis=1,inplace=True)
data_new=pd.concat([data,X1],axis=1)
data_new
X=data_new.iloc[:,1:11]
X.shape
list(X.iloc[:,:7])
X.iloc[:,:7].hist()
sns.distplot(X.iloc[:,:7])


Y=data_new['Sales']

Y1=[]
#Converting Y variable to Categorival format
for i in range(0,400,1):
    if Y.iloc[i,]>=Y.mean():
        print('High')
        Y1.append('High')
    else:
        print('Low')
        Y1.append('Low')
Y_new=pd.DataFrame(Y1)

#Preprocessing the data
from sklearn.preprocessing import StandardScaler, LabelEncoder
SS=StandardScaler()
LE=LabelEncoder()
X.iloc[:,:7]=SS.fit_transform(X.iloc[:,:7])
for i in range(7,10,1):
    X.iloc[:,i]=LE.fit_transform(X.iloc[:,i])
print(X)
X.head()

#Splitting the data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y_new,test_size=0.25,stratify=Y_new,random_state=91)
X_train.shape

#Fitting the model
from sklearn.tree import DecisionTreeClassifier
DT=DecisionTreeClassifier(criterion='entropy',max_depth=8).fit(X_train,Y_train)
Y_pred=DT.predict(X_test)

#Metrics
from sklearn.metrics import accuracy_score
acc=accuracy_score(Y_test, Y_pred)
print(acc)

#Tree
import matplotlib.pyplot as plt
from sklearn import tree
tr=tree.plot_tree(DT,filled=True,fontsize=6)

DT.tree_.node_count
DT.tree_.max_depth
'''
Inference: For random state 91, and for the max depth of 8 in decision tree we are achieving 
           the 80% accuracy for the given dataset.
'''