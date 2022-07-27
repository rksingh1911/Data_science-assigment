# -*- coding: utf-8 -*-
"""
Created on Sun May 22 19:29:25 2022

@author: Suklesh
"""
#Random Forest on fraud data
import pandas as pd
import seaborn as sns
import numpy as np
data=pd.read_csv('F:\\ExcelR\\Assignments\\Random Forests\\Fraud_check.csv')
list(data)
data.dtypes

X1=data[['Undergrad','Marital.Status','Urban']]
X2=data[['City.Population','Work.Experience']]

#Countplot for categorical variables
sns.countplot(data['Undergrad'])
sns.countplot(data['Marital.Status'])
sns.countplot(data['Urban'])

#Histograms for numerical variables
sns.distplot(data['Taxable.Income'])
sns.distplot(data['City.Population'])
sns.distplot(data['Work.Experience'])

#Data preprocessing
from sklearn.preprocessing import LabelEncoder,StandardScaler
LE=LabelEncoder()
SS=StandardScaler()

for i in range(0,3,1):
    X1.iloc[:,i]=LE.fit_transform(X1.iloc[:,i])
    
X_scale=SS.fit_transform(X2)
X_scale=pd.DataFrame(X_scale)
X_scale.set_axis(['City.Population','Work.Experience'],axis='columns',inplace=True)

X_new=pd.concat([X1,X_scale],axis=1)
X_new
X_new.shape

'''
#As per the problem statement we are asked to convert this Y variable into categorical
#So, Differentiating the Y variable with respect to 30000 (as mentioned in the prob statement)
#Sales greater than or equal to mean is categorised as High, otherwise Low
'''
#Target variable
Y=data['Taxable.Income']
data.drop(['Taxable.Income'],axis=1,inplace=True)
data.describe()

#Converting Y variable into categorical 
Y1=[]
for i in range(0,600,1):
    if Y.iloc[i,]<=30000:
        print('Risky')
        Y1.append('Risky')
    else:
        print('Good')
        Y1.append('Good')


Y_new=pd.DataFrame(Y1)
Y_new[0].ndim
list(Y_new)
Y_new.set_axis(['Taxable.Income'],axis='columns',inplace=True)
sns.countplot(Y_new['Taxable.Income'])

Y_new.ndim
Y_new['Taxable.Income'].ndim


#Splitting into train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X_new,Y_new['Taxable.Income'],stratify=Y_new['Taxable.Income'],test_size=0.25,random_state=67)
X_train.shape


Y_new.ndim

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
RFC=RandomForestClassifier(max_features=0.3,n_estimators=500)
model=RFC.fit(X_train,Y_train)
Y_pred=model.predict(X_test)

#Metrics
from sklearn import metrics
from sklearn.metrics import accuracy_score,mean_squared_error
acc=accuracy_score(Y_test, Y_pred)
print(acc)

acc1=[]
set1=np.arange(0.1,1.1,0.1)
for i in set1:
    RFC=RandomForestClassifier(max_features=i,n_estimators=500)
    model = RFC.fit(X_train,Y_train)
    Y_pred=model.predict(X_test)
    acc=accuracy_score(Y_test, Y_pred)
    acc1.append((acc*100).round(3))
    print('For max features',i,',accuracy is',(acc*100).round(3))
    
import matplotlib.pyplot as plt
plt.plot(set1,acc1,data=None)
plt.xlabel('max_features')
plt.ylabel('accuracy')
plt.title('Graph between max features and accuracy')
plt.show()

'''
Inference: By the plot we can understand that by considering 30% of the columns we can get best accuracy
           of 79.33% for the given data at the random state 0f 67.
'''
for i in range(1,101,1):
    X_train,X_test,Y_train,Y_test=train_test_split(X_new,Y_new['Taxable.Income'],stratify=Y_new,test_size=0.25,random_state=i)
    print('For random state',i)
    for j in np.arange(0.1,1.1,0.1):
        RFC=RandomForestClassifier(max_features=j,n_estimators=500)
        model=RFC.fit(X_train,Y_train)
        Y_pred=model.predict(X_test)
        acc=accuracy_score(Y_test, Y_pred)
        while (acc*100)>=77:
            print('Accuracy for max_features',j,'is',(acc*100).round(3))
            break


#Y_train.values.ravel(

