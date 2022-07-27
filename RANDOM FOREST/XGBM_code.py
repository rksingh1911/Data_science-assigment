# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 16:23:09 2022

@author: Hi
"""

# First XGBoost model for Pima Indians dataset
#pip install xgboost
from xgboost import XGBClassifier

# load data
import pandas as pd
dataset = pd.read_csv("pima-indians-diabetes.data.csv")
# split data into X and y
X = dataset.iloc[:,0:8]
Y = dataset.iloc[:,8]

dataset
dataset.shape

# split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)


# fit model no training data
#eta=0.001,gamma=10,learning_rate=1
XGBClassifier()
# model = XGBClassifier(n_estimators=90,max_depth=3,eta=0.001,gamma=10,learning_rate=1)
model = XGBClassifier(n_estimators=100,eta=0.001,gamma=10,learning_rate=0.5)
# model = XGBClassifier()
model.fit(X_train, y_train)


# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]


# evaluate predictions
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

