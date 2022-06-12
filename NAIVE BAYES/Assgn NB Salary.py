# -*- coding: utf-8 -*-
"""
Created on Mon May  9 13:35:56 2022

@author: Suklesh
"""
#Navie Bayes on Salary data
#As we are provided with 2 datasets as train and test, I am using the train data to fit the model
#and test data for prediction

import pandas as pd
import seaborn as sns
data_train=pd.read_csv('F:\\ExcelR\\Assignments\\Navie Bayes\\SalaryData_Train.csv')
data_train.shape
list(data_train)
data_test=pd.read_csv('F:\\ExcelR\\Assignments\\Navie Bayes\\SalaryData_Test.csv')
data_test.shape
list(data_test)

#Splitting the data into train and test from respective datasets
#Training data
X_train=data_train.iloc[:,:13]
list(X_train)
X_train.dtypes
X_tr_1=data_train[['age','educationno','capitalgain','capitalloss','hoursperweek']] #Numerical data in X_train
X_tr_1.hist()

X_tr_2=data_train[['workclass','education','maritalstatus','occupation','relationship','race','sex','native']] #Categorical data in X_train
sns.countplot(X_tr_2['workclass'])
sns.countplot(X_tr_2['education'])
sns.countplot(X_tr_2['maritalstatus'])
sns.countplot(X_tr_2['occupation'])
sns.countplot(X_tr_2['relationship'])
sns.countplot(X_tr_2['race'])
sns.countplot(X_tr_2['sex'])

X_tr_2.shape
Y_train=data_train['Salary']

#Test data
X_test=data_test.iloc[:,:13]
list(X_test)
X_test.dtypes
X_te_1=data_test[['age','educationno','capitalgain','capitalloss','hoursperweek']] #Numerical data in X_test
X_te_2=data_test[['workclass','education','maritalstatus','occupation','relationship','race','sex','native']] #Categorical data in X_test
Y_test=data_test['Salary']

#Converting the categorical data of both X_train and X_test i.e, X_tr_2,X_te_2
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
for i in range(0,8,1):
    X_tr_2.iloc[:,i]=LE.fit_transform(X_tr_2.iloc[:,i])
    X_te_2.iloc[:,i]=LE.fit_transform(X_te_2.iloc[:,i])
print(X_tr_2)
list(X_tr_2)
X_tr_2.head()
print(X_te_2)


#Considering the processed data into new variable
X_train_new=pd.concat([X_tr_1,X_tr_2],axis=1)
X_test_new=pd.concat([X_te_1,X_te_2],axis=1)

#Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
MNB=MultinomialNB().fit(X_train_new,Y_train)
Y_pred=MNB.predict(X_test_new)

#Metrics
from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(Y_test,Y_pred)
print(cm)
acc=accuracy_score(Y_test,Y_pred)
print(acc)

#Bernoulli Naive 
from sklearn.naive_bayes import BernoulliNB
BNB=BernoulliNB().fit(X_train_new,Y_train)
Y_pred1=BNB.predict(X_test_new)

#Metrics
cm1=confusion_matrix(Y_test,Y_pred1)
print(cm1)
acc1=accuracy_score(Y_test,Y_pred1)
print(acc1)

'''
Inference: Here we are given with the data which is already spilt into train and test
           Converting the respective categorical variable under data preprocessing and fitting the model
           We did not standardise the numerical data here as the naive bayes is based on probability concept
           where we cannot accept negative values

Best fit model is using Multinomial Naive Bayes with the accuracy of 77.49%
'''

           