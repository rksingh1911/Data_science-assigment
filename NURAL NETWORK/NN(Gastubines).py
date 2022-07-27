"""
Created on Sat Jun 25 15:12:42 2022

@author: ROHIT SINGH
"""
#Business Problem
"predicting turbine energy yield (TEY) using ambient variables as features."

#Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#loading the data
data=pd.read_csv("E:\\ExcelR\\assigment\\NURAL NETWORK\\gas_turbines.csv")
data.shape
list(data)
data.dtypes

Y=data['TEY']
sns.distplot(Y)
data.drop(['TEY'],axis=1,inplace=True)
data.shape
list(data)

X=data.iloc[:,0:3]
list(X)
X.hist()

#Loading the model
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
model=Sequential()
model.add(Dense(9,input_dim=3,activation='relu'))
model.add(Dense(1,activation='relu'))

#Compiling the model
model.compile(loss='msle',optimizer='adam',metrics=['msle'])

#Fitting the model
fv=model.fit(X,Y,validation_split=0.25,epochs=250,batch_size=10)

#List of data in history
fv.history.keys()

#Evaluate the model
scores=model.evaluate(X,Y)
print('%s:%.2f%%'%(model.metrics_names[1],scores[1]))

#Summary of history for accuracy
plt.plot(fv.history['msle'])
plt.plot(fv.history['val_msle'])
plt.title('model msle')
plt.xlabel('epochs')
plt.ylabel('msle')
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