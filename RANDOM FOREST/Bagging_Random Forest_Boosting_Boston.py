# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:32:45 2022

@author: Hi
"""

"""
CRIM - per capita crime rate by town
ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
INDUS - proportion of non-retail business acres per town.
CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
NOX - nitric oxides concentration (parts per 10 million)
RM - average number of rooms per dwelling
AGE - proportion of owner-occupied units built prior to 1940
DIS - weighted distances to five Boston employment centres
RAD - index of accessibility to radial highways
TAX - full-value property-tax rate per $10,000
PTRATIO - pupil-teacher ratio by town
B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
LSTAT - % lower status of the population
MEDV - Median value of owner-occupied homes in $1000's

"""
###############################################################################
import numpy as np
import pandas as pd
df = pd.read_csv('D:\\CARRER\\My_Course\\Data Science Classes\\3 Module\\1 Supervised\\11 Ensemble Methods\\1 Random forests\\Boston.csv')
df.head()
df.shape


X = df.iloc[:,1:14] # Only Independent variables
X.shape
X.head()

Y = df.medv
###############################################################################

# Splitting train and test data sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

X_train.shape
X_test.shape
Y_train.shape
Y_test.shape


######################### Bagging Regressor ###################################
# n_estimators = 500 trees 
# bagging Regressor # max_samples = 50% of samples for every tree
from sklearn.tree import DecisionTreeRegressor # Classifier
DT = DecisionTreeRegressor() 

from sklearn.ensemble import BaggingRegressor # Classifier--> BaggingClassifier
Bag = BaggingRegressor(base_estimator = DT ,max_samples = 0.8 ,n_estimators = 500, random_state = 8) 
Bag.fit(X_train,Y_train)
Y_pred = Bag.predict(X_test)

from sklearn import metrics  
np.sqrt(metrics.mean_squared_error(Y_pred,Y_test)).round(3)

###############################################################################
# Create two lists for training and test errors
training_Error = []
test_Error = []

# Define a range of 1 to 10 (included) neighbors to be tested
settings = np.arange(0.1, 1.1, 0.1)

# Loop with the  DecisionTreeRegressor through the Max depth values to determine the most appropriate (best)
from sklearn.ensemble import BaggingRegressor # Classifier
from sklearn import metrics  

for samp_val in settings:
    regressor = BaggingRegressor(base_estimator = DT , n_estimators=500,random_state=42,max_samples=samp_val)
    regressor.fit(X_train, Y_train)

    Y_Train_pred = regressor.predict(X_train)
    training_Error.append(np.sqrt(metrics.mean_squared_error(Y_Train_pred, Y_train).round(3)))

    Y_Test_pred = regressor.predict(X_test)
    test_Error.append(np.sqrt(metrics.mean_squared_error(Y_Test_pred, Y_test).round(3)))

print(training_Error)
print(test_Error)




# Visualize results - to help with deciding which n_neigbors yields the best results (n_neighbors=6, in this case)
import matplotlib.pyplot as plt
plt.plot(settings, training_Error, label='RMSE of the training set')
plt.plot(settings, test_Error, label='RMSE of the test set')
plt.ylabel('Root Mean Squared Error')
plt.xlabel('Percentage of sample in Bag')
plt.legend()

###############################################################################

################## RANDOM FORESTS #################################

from sklearn.ensemble import RandomForestRegressor # Classifier
RF = RandomForestRegressor(max_features = 0.5,n_estimators = 500, random_state = 8) 
RF.fit(X_train,Y_train)
Y_pred = RF.predict(X_test)

from sklearn import metrics  
np.sqrt(metrics.mean_squared_error(Y_pred,Y_test)).round(3)

###############################################################################

###############################################################################
# Create two lists for training and test errors
training_Error = []
test_Error = []

# Define a range of 1 to 10 (included) neighbors to be tested
settings = np.arange(0.1, 1.1, 0.1)

# Loop with the  DecisionTreeRegressor through the Max depth values to determine the most appropriate (best)
from sklearn.ensemble import RandomForestRegressor # Classifier
from sklearn import metrics  

for samp_val in settings:
    regressor = RandomForestRegressor(n_estimators=500,random_state=42,max_features=samp_val)
    regressor.fit(X_train, Y_train)

    Y_Train_pred = regressor.predict(X_train)
    training_Error.append(np.sqrt(metrics.mean_squared_error(Y_Train_pred, Y_train).round(3)))

    Y_Test_pred = regressor.predict(X_test)
    test_Error.append(np.sqrt(metrics.mean_squared_error(Y_Test_pred, Y_test).round(3)))

print(training_Error)
print(test_Error)

# Visualize results - to help with deciding which n_neigbors yields the best results (n_neighbors=6, in this case)
import matplotlib.pyplot as plt
plt.plot(settings, training_Error, label='RMSE of the training set')
plt.plot(settings, test_Error, label='RMSE of the test set')
plt.ylabel('Root Mean Squared Error')
plt.xlabel('Percentage of features in RF')
plt.legend()


