# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 14:27:53 2022

@author: ROHIT SINGH
"""

#Business Problem
"PREDICT THE BURNED AREA OF FOREST FIRES WITH NEURAL NETWORKS"

#Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#loading the data
data=pd.read_csv("E:\\ExcelR\\assigment\\NURAL NETWORK\\forestfires.csv")
data.shape
list(data)
data.dtypes

Y=data['size_category']
sns.countplot(Y)
data.drop(['size_category'],axis=1,inplace=True)
data.shape
list(data)

X=data.iloc[:,2:30]
list(X)
X.shape
X.iloc[:,0:7].hist()

#Preprocessing the target variable
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
Y_scale=LE.fit_transform(Y)
Y_scale

#Loading the model
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
model=Sequential()
model.add(Dense(42,input_dim=28,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#Compiling the model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#Fitting the model
fv=model.fit(X,Y_scale,validation_split=0.25,epochs=250,batch_size=10)

#List of data in history
fv.history.keys()

#Evaluate the model
scores=model.evaluate(X,Y_scale)
print('%s:%.2f%%'%(model.metrics_names[1],scores[1]*100))

#Summary of history for accuracy
plt.plot(fv.history['accuracy'])
plt.plot(fv.history['val_accuracy'])
plt.title('model accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train','test'],loc='upper right')
plt.show()

#Summary of history for loss
plt.plot(fv.history['loss'])
plt.plot(fv.history['val_loss'])
plt.title('model loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train','test'],loc='upper right')
plt.show()
