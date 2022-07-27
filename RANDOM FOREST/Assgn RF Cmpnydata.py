# -*- coding: utf-8 -*-
"""
Created on Sun May  1 11:52:22 2022

@author: Suklesh
"""
#Random Forest on company data
import pandas as pd
import seaborn as sns
data=pd.read_csv("F:\\ExcelR\\Assignments\\Random Forests\\Company_Data (1).csv")
data.shape
list(data)
data.dtypes
X1=data['ShelveLoc']
X1.head
data.drop(['ShelveLoc'],axis=1,inplace=True)
data_new=pd.concat([data,X1],axis=1)
list(data_new)

X=data_new.iloc[:,1:11]
list(X)
X.dtypes
X.iloc[:,0:7].hist()
sns.countplot(x ='Urban', data = data_new)
sns.countplot(x ='US', data = data_new)
sns.countplot(x ='ShelveLoc', data = data_new)

Y=data_new['Sales']
Y.shape
Y_mean=Y.mean()
'''
#As per the problem statement we are asked to convert this Y variable into categorical
#So, Differentiating the Y variable with respect to mean
#Sales greater than or equal to mean is categorised as High, otherwise Low
'''
#Converting Y variable into categorical 
Y1=[]
for i in range(0,400,1):
    if Y.iloc[i,]>=Y_mean:
        print('High')
        Y1.append('High')
    else:
        print('Low')
        Y1.append('Low')

Y_new=pd.DataFrame(Y1)
list(Y_new)
Y_new.set_axis(['target'],axis='columns',inplace=True)
sns.countplot(Y_new['target'])
#Preprocessing the data
from sklearn.preprocessing import LabelEncoder,StandardScaler
LE=LabelEncoder()        
Y_new=LE.fit_transform(Y_new)
Y_new

#Preprocessing the data
from sklearn.preprocessing import LabelEncoder,StandardScaler
LE=LabelEncoder()        
SS=StandardScaler()
X['Urban']=LE.fit_transform(X['Urban'])
X['US']=LE.fit_transform(X['US'])
X['ShelveLoc']=LE.fit_transform(X['ShelveLoc'])
X_scale=SS.fit_transform(X.iloc[:,0:6])
X_scale=pd.DataFrame(X_scale)
X_scale.set_axis(['CompPrice','Income','Advertising','Population','Price','Age'],axis='columns',inplace=True)
X_new=pd.concat([X_scale,X['Urban'],X['US'],X['ShelveLoc']],axis=1)
X_new

#Splitting into train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X_new,Y_new,stratify=Y_new,test_size=0.25,random_state=37)
X_train.shape

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
RFC=RandomForestClassifier(max_features=0.4,n_estimators=500)
model=RFC.fit(X_train,Y_train)
Y_pred=model.predict(X_test)

#Metrics
from sklearn import metrics
from sklearn.metrics import accuracy_score,mean_squared_error
acc=accuracy_score(Y_test, Y_pred)
print(acc)

'''
Checking the training error and test error by creating a empty list respectively
'''
import numpy as np
tr_err=[]
t_err=[]
set1=np.arange(0.1,1.1,0.1)
for j in set1:
    RFC=RandomForestClassifier(max_features=j,n_estimators=500)
    model=RFC.fit(X_train,Y_train)
        
    Y_pred_tr=model.predict(X_train)
    Y_pred_te=model.predict(X_test)
        
    tr_err.append(np.sqrt(metrics.mean_squared_error(Y_train,Y_pred_tr)))
    t_err.append(np.sqrt(metrics.mean_squared_error(Y_test,Y_pred_te)))
    

TR_err=np.mean(tr_err)
TE_err=np.mean(t_err)

print(TR_err)
print(TE_err)

'''
Training error is shown as zero as we use the training data for fitting the model
'''

import matplotlib.pyplot as plt
plt.plot(set1,tr_err,label='Training error')
plt.plot(set1,t_err,label='Test error')
plt.xlabel('No of features')
plt.ylabel('Error')
plt.title('Graph')
plt.show()
'''
From the graph we can observe that for max_features of 0.4 i.e, considering the 40% of the columns
we can see the minimum test error, so taking it to fit the model
'''        

    
    



