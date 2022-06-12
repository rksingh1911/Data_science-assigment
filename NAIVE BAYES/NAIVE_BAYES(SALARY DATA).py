"""
Created on Tue Jun  7 13:28:03 2022

@author: ROHIT SINGH
"""

#Business Problem
"Prepare a classification model using Naive Bayes for salary data"
"we have to preper a model for salary data set so we can use train data to fit and test data for predictinn"


#Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

#loading the data
df_train = pd.read_csv("E:\\ExcelR\\assigment\\NAIVE BAYES\\SalaryData_Train.csv")
df_test = pd.read_csv("E:\\ExcelR\\assigment\\NAIVE BAYES\\SalaryData_Test.csv")

#Exploratory Data Analysis
df_train.head
df_test.head
df_train.tail
df_test.tail
df_train.shape
df_test.shape
df_train.ndim
df_test.ndim
type(df_train)
type(df_test)
list(df_train)
list(df_test)

df_train.info()
df_test.info()
#so here we have int and char both data we have to chenge it

#data cleaning

#mapping terget variable

mapping = {' >50K': 0, ' <=50K': 1}
df1_train = df_train.replace({'Salary': mapping})
y1_train =  df1_train['Salary']
df1_test = df_test.replace({'Salary': mapping})
y1_test = df1_test['Salary']
#saperating the  categorical data
x1_train_cat = df1_train[['workclass','education','maritalstatus','occupation','relationship','race','sex','native']]
x1_train_cat.count()
x1_train_cat.shape
x1_test_cat = df1_test[['workclass','education','maritalstatus','occupation','relationship','race','sex','native']]
x1_test_cat.count()
x1_test_cat.shape

#Lable Encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in range (0,8,1):
    x1_train_cat.iloc[:,i] = le.fit_transform(x1_train_cat.iloc[:,i])
    x1_test_cat.iloc[:,i] = le.fit_transform(x1_test_cat.iloc[:,i])
print(x1_train_cat)
print(x1_test_cat)

#saperating the  continouse data
x1_train_con = df1_train[['age','educationno','capitalgain','capitalloss','hoursperweek']]                   
x1_test_con = df1_test[['age','educationno','capitalgain','capitalloss','hoursperweek']]

#standardzation                    
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x1_train_std = scaler.fit_transform(x1_train_con)
x1_train_dummy = pd.DataFrame(x1_train_std)
x1_train = x1_train_dummy.set_axis(['age','educationno','capitalgain','capitalloss','hoursperweek'],axis=1)


x1_test_std = scaler.fit_transform(x1_test_con)
x1_test_dummy = pd.DataFrame(x1_test_std)
x1_test = x1_test_dummy.set_axis(['age','educationno','capitalgain','capitalloss','hoursperweek'],axis=1)
                    
#Considering the processed data into new variable
df_train_new = pd.concat([x1_train,x1_train_cat,y1_train],axis= 1,sort=True)
df_train_new.shape
list(df_train_new)
df_test_new = pd.concat([x1_test,x1_test_cat,y1_test],axis=1)
df_test_new.shape
list(df_test_new)

#describe for all mean mode information
df_train_new.describe()
df_test_new.describe()

#finding null value
df_train_new.isnull()
df_test_new.isnull()

#skew for normal distributaion
df_train_new.skew()  #so  only capitalgain and capitalloss variable skewness are very high
df_train_new.skew()

#corelation
df_train_new.corr()
df_train_new.corr()
'''As seen in the above graph, there is a high correlation 
exists between some of the variables. 
We can use PCA to reduce the hight correlated variables'''

#Visualisation on data
df_train_new.hist(figsize = (20,10))
df_test_new.hist(figsize = (20,10))

#countplot
sns.countplot(x = "Salary",data = df_train_new)
sns.countplot(x = "Salary",data = df_test_new)
#As shown in the graphs above, highest number of people available in data are getting more then 50K 

sns.countplot(x = "sex", data = df_train_new)
sns.countplot(x = "sex",data = df_test_new)
#most of the salary taker above 50K are male

sns.countplot(x = "maritalstatus", data = df_train_new)
sns.countplot(x = "maritalstatus",data = df_test_new)
#most of them are Married-chi-spouse and saprated

sns.countplot(x = "race", data = df_train_new)
sns.countplot(x = "race",data = df_test_new)
#most of them are white

pd.crosstab(df_train_new.workclass,df_train_new.Salary).plot(kind = 'bar')
pd.crosstab(df_test_new.workclass,df_test_new.Salary).plot(kind = 'bar')
#most of the age of the people who getting <50k they are private job

pd.crosstab(df_train_new.age,df_train_new.Salary).plot(kind = 'line')
pd.crosstab(df_test_new.age,df_test_new.Salary).plot(kind = 'line')
#most of the age of the people who getting <50k they are between .02 to .06

#density plot 
df_train_new.plot(kind='density', subplots=True, layout=(4,5), figsize=(13,20), sharex=False, sharey=False)
plt.show()
df_test_new.plot(kind = 'density', subplots =True, layout=(4,5),figsize = (13,20),sharex = False, sharey = False)
plt.show()

#barplot
df_train_new.plot(kind = 'bar',subplots=True, layout=(4,5), figsize=(13,20))
df_test_new.plot(kind = 'bar',subplots = True,layout = (4,5),figsize = (13,20))

#line plot
df_train_new.plot(kind = 'line',figsize = (13,20))
df_test_new.plot(kind = 'line',figsize = (13,20))

#pair plot
sns.pairplot(df_train_new)
sns.pairplot(df_test_new)

#Naive Bayes

# split as X and Y vairables for train data set
x_train = df_train_new.iloc[:,0:13]
x_train.shape
x_train.ndim
y_train = df_train_new['Salary']
# split as X and Y vairables for test data set
x_test = df_test_new.iloc[:,0:13]
x_test.shape
x_test.ndim
y_test = df_test_new['Salary']

#normality
#Test of hypothesis 
from scipy.stats import shapiro
stat, p = shapiro(x_train)
print(stat)
print("p-value",p)

alpha = 0.05 # 5% level of significance

if p < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")

# H1: Data is normal
"we are checking only for train data set because we have to fit model in train data"
#Multinomial Naive Bayes Algorithms
from sklearn.naive_bayes import MultinomialNB
MNB=MultinomialNB().fit(x_train,y_train)
Y_pred = MNB.predict(x_test)

#Metrics
from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(y_test,Y_pred)
print(cm)
acc=accuracy_score(y_test,Y_pred)
print(acc)

#Bernoulli Naive 
from sklearn.naive_bayes import BernoulliNB
BNB=BernoulliNB().fit(x_train,y_train)
Y_pred1=BNB.predict(x_test)

#Metrics
cm1=confusion_matrix(y_test,Y_pred1)
print(cm1)
acc1=accuracy_score(y_test,Y_pred1)
print(acc1)

#Gaussian Naive
from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB().fit(x_train,y_train)
Y_pred2 = GNB.predict(x_test)

#Metrics
cm2=confusion_matrix(y_test,Y_pred2)
print(cm2)
acc2=accuracy_score(y_test,Y_pred2)
print(acc2)

#Conclusion & Cross Validation
"GaussianNB Model has a better Accuracy 79%, Thus we will use GaussianNB Classifier"

#We will also cross validate the model with other classifiers to get better understanding of which classifier is best suited for our data

#import other algorithems
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import model_selection
#creat empty set
models = []
result = []
names = []
score = "accuracy"

#models
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('GB', GaussianNB()))
models.append(('NB',MultinomialNB()))
models.append(('BB',BernoulliNB()))

for name ,model in models:
    kfold = model_selection.KFold(n_splits = 10)
    cv_results = model_selection.cross_val_score(model,x_train,y_train,cv = kfold, scoring = score)
    result.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" %(name, cv_results.mean(),cv_results.std())
    print(msg)
    
# In comparision KNN has the best Accuracy





