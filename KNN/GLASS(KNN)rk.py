"""
Created on Mon Jun  6 01:27:05 2022

@author: ROHIT SINGH
"""

##Business Problem
"Prepare a model for glass classification using KNN"

#Importing the libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

#loading the data
df = pd.read_csv("E:\\ExcelR\\assigment\\KNN\\glass.csv")

#Exploratory Data Analysis
df.head()
df.tail()
df.shape
type(df)
list(df)
df.ndim
df.info()

#skew for normal distributaion
df.skew()     #for all variable skewness in very high

#describe for all mean mode information
df.describe()

#corelation
df.corr()
'''As seen in the above , there is a avg correlation 
exists between some of the variables. '''

#finding null value
df.isnull()

#Visualisation on data
df.hist(figsize = (20,9))

#countplot
sns.countplot(x = 'Type',data = df)
#As shown in the graphs above, highest number of glass available that are Type 2 followed by 1,7 and 3 respectively

#Stackedbarchart&unstackedbarchart
df.plot.bar(figsize=(15, 8))
df.plot.bar(stacked = True,figsize=(15, 8))
#as shown in the graphs above ,we can understand that SI and NA CA has higest waighage

#bar graphs Si vs type
pd.crosstab(df.Si,df.Type).plot(kind = 'bar')
#we can see si is use equaliy to make all type of glass

#bar graphs Na vs Type
pd.crosstab(df.Na,df.Type).plot(kind = 'bar')
#we can see Na is use more to make 2 type of glass

#graphs Ca vs type
pd.crosstab(df.Ca,df.Type).plot(kind = 'bar')
#we can see ca is use more to make 1 2 and 5 type of glass

#grphs RI vs type
pd.crosstab(df.RI,df.Type).plot(kind = 'bar')
#we can see RI is use in a littile amount for all type of glass

#graphs Al vs type
pd.crosstab(df.Al,df.Type).plot(kind = 'bar')
#we can see it is use to make 1 and 2 type of glass more

#graphs K vs type
pd.crosstab(df.K,df.Type).plot(kind = 'bar')
#we can see it is use to equaliy us

#graph BA vs type
pd.crosstab(df.Ba,df.Type).plot(kind = 'bar')
#we can see is only use for 5 type of glass

#graphs Fe vs type
pd.crosstab(df.Fe,df.Type).plot(kind = 'bar')
# we can see it is use to make 1 type of glass

#density plot 
df.plot(kind='density', subplots=True, layout=(4,5), figsize=(13,20), sharex=False, sharey=False)
plt.show()
#we can see for all variable how data is distibutaed for glass

#borplot
df.plot(kind = 'bar',subplots = True, layout = (4,5),figsize = (20,15),sharex = False,sharey = False)
#so we can observ that we in our dataset si,ca ,na,mg are useed most for glass

#boxplot
df.plot(kind = 'box',subplots = True, layout = (4,5),figsize = (20,15),sharex = False,sharey = False)
#we can see in our data have how much outliers

#line plot
df.plot(kind = 'line',figsize = (13,20),)

#pair plot
sns.pairplot(df)

#DATA cleaning
df.isnull().sum()
#so our data is very clean

#KNN clasifier  Model

# split as X and Y vairables
x = df.iloc[:,0:9]
x.shape
x.ndim
y = df['Type']

#crosstablte
pd.crosstab(y,y)

#Splitting data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,stratify = y,random_state= 1)


# Install KNN
from sklearn.neighbors import KNeighborsClassifier

#create empty set
train_accuracy = []
test_accuracy = []

#here big task is to find K value so for that we ll take one range 
k_values = np.arange(1,25)

for i, k in enumerate(k_values):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train,y_train)
    train_accuracy.append(knn.score(x_train,y_train))
    test_accuracy.append(knn.score(x_test,y_test))
    
#plot line graph for k value better understaing
plt.Figure(figsize=[20,8])
plt.plot(k_values,train_accuracy,label = 'training accuracy')
plt.plot(k_values,test_accuracy,label = 'testing accuracy')
plt.legend()
plt.title('-value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.show()    
#As shown in the graph, with K=4 we can achive accurary of 70%.

#Applying the algorithm
knn = KNeighborsClassifier(n_neighbors = 4).fit(x_train,y_train)

# Make predictions using independent variable values
Y_pred = knn.predict(x_test)

# Compute confusion matrix
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, Y_pred)
print(cm)

#accuracy score
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,Y_pred)
print(acc)


#Cross validation
traning_error = []
test_error = []
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

for i in range(0,101,1):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state = i)
    knn = KNeighborsClassifier(n_neighbors = 2).fit(x_train,y_train)
    
    Y_pred_traning = knn.predict(x_train)
    Y_pred_test = knn.predict(x_test)
    
    traning_error.append(accuracy_score(y_train, Y_pred_traning))
    test_error.append(accuracy_score(y_test,Y_pred_test))
    
    traning_error = np.mean(traning_error)
    test_error = np.mean(test_error)
    print(traning_error)
    print(test_error)

#Conclusion
"so we can see in traning error and test error we dont have not much more dinfference" 
#accuracy = 68% 

