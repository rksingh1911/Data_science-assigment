"""
Created on Sun Jun  5 13:05:32 2022

@author: ROHIT SINGH
"""

#Business Problem
"Implement a KNN model to classify the animals in to categorie"


#Importing the libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

#loading the data
df = pd.read_csv("E:\\ExcelR\\assigment\\KNN\\zoo.csv")

#Exploratory Data Analysis
df.head()
list(df)
type(df)
df.ndim
df.info()
df.shape

#skew for normal distributaion
df.skew()  #for all variable skewness in very high

#describe for all mean mode information
df.describe()

#corelation
df.corr()
'''As seen in the above graph, there is a high correlation 
exists between some of the variables. 
We can use PCA to reduce the hight correlated variables'''

#finding null value
df.isnull()

#Visualisation on data
df.hist(figsize= (20,7))

#countplot
sns.countplot(x = "type",data = df)
#As shown in the graphs above, highest number of animals available in Zoo are Type 1 followed by 2, 4 and 7 respectively

sns.countplot(x = "feathers",data = df)
#As shown in the graphs above, highest number of animals available who have feathers

sns.countplot(x = "aquatic",data = df)
#As shown in the graphs above, equal number of animals available who can live in water and land

sns.countplot(x = "toothed",data = df)
#As shown in the graphs above, highest number of animals available who have tooth

sns.countplot(x = "legs",data = df)
#As shown in the graphs above, highest number of animals available who have 4 legs

sns.countplot(x = "tail",data = df)
#As shown in the graphs above, highest number of animals available tail

sns.countplot(x = "domestic",data = df)
#As shown in the graphs above, highest number of animals available those are no domestic

#density plot 
df.plot(kind='density', subplots=True, layout=(4,5), figsize=(13,20), sharex=False, sharey=False)
plt.show()
#we can see for all variable how data is distibutaed for animal

#barplot
df.plot(kind = 'bar',subplots=True, layout=(4,5), figsize=(13,20),)

#line plot
df.plot(kind = 'line',figsize = (13,20),)

#pair plot
sns.pairplot(df)

#DATA cleaning
df.isnull().sum()
df1 = df.drop('animal name',axis = 1)
print(df1)

#KNN clasifier  Model

# split as X and Y vairables
x = df1.iloc[:,0:16]
x.shape
x.ndim
y = df1['type']

#crosstablte
pd.crosstab(y,y)

#Splitting data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2 ,stratify=y,random_state= 1)

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
#As shown in the graph, with K=5 we can achive accurary of 92%.

#Applying the algorithm
knn = KNeighborsClassifier(n_neighbors = 5).fit(x_train,y_train)

# Make predictions using independent variable values
Y_pred = knn.predict(x_test)

# Compute confusion matrix
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, Y_pred)
print(cm)

#accuracy score
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,Y_pred)
print(acc)

#Cross validation
traning_error = []
test_error = []
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

for i in range(0,101,1):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = i)
    knn = KNeighborsClassifier(n_neighbors = 5).fit(x_train,y_train)
    
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
#accuracy = 77%    