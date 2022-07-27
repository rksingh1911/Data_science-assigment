# -*- coding: utf-8 -*-
"""
Created on Tue May 24 15:38:02 2022

@author: Suklesh
"""
#SVM on forest fire dataset
import pandas as pd
data=pd.read_csv('F:\\ExcelR\\Assignments\\Support Vector machine\\forestfires.csv')
data.shape
list(data)
data.dtypes

#Data preprocessing
#Data which require standardization
X1=data[['FFMC','DMC','DC','ISI','temp','RH','wind','rain','area']]
from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
X1_scale=SS.fit_transform(X1)
X1_new=pd.DataFrame(X1_scale)
X1_new.set_axis(['FFMC','DMC','DC','ISI','temp','RH','wind','rain','area'],axis='columns',inplace=True)
X1_new.head()

#Data which require Label encoding
X2=data[['month','day','size_category']]
#X2_scale=pd.DataFrame()

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
for i in range(0,3,1):
    X2.iloc[:,i]=LE.fit_transform(X2.iloc[:,i])

Y=X2['size_category']
X2.drop(['size_category'],axis=1,inplace=True)

X3=data[['dayfri','daymon','daysat','daysun','daythu','daytue','daywed',
         'monthapr','monthaug','monthdec','monthfeb','monthjan','monthjul',
         'monthjun','monthmar','monthmay','monthnov','monthoct','monthsep']]

#Combining all the X data
X=pd.concat([X1_new,X2,X3],axis=1)

#Splitting into train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=4)
X_train.shape

#Loading svc (Linear fitting)
from sklearn.svm import SVC
svl=SVC(kernel='linear').fit(X_train,Y_train)
Y_pred=svl.predict(X_test)

#Metrics
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(Y_test,Y_pred)
print(cm)
acc=accuracy_score(Y_test,Y_pred)
print(acc)

#Loading svc (Radial bias fitting)
from sklearn.svm import SVC
svr=SVC(kernel='rbf').fit(X_train,Y_train)
Y_pred1=svr.predict(X_test)

#Metrics
from sklearn.metrics import confusion_matrix,accuracy_score
cm1=confusion_matrix(Y_test,Y_pred1)
print(cm1)
acc1=accuracy_score(Y_test,Y_pred1)
print(acc1)

#Loading svc (Polynomial)
from sklearn.svm import SVC
svp=SVC(kernel='poly',degree=3).fit(X_train,Y_train)
Y_pred2=svp.predict(X_test)

#Metrics
from sklearn.metrics import confusion_matrix,accuracy_score
cm2=confusion_matrix(Y_test,Y_pred2)
print(cm2)
acc2=accuracy_score(Y_test,Y_pred2)
print(acc2)
'''
Inference: From above results we can see that the best fit is by using linear kernel
           and accuracy achieved is around 94%
'''

    