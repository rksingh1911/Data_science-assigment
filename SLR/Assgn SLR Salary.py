  # -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 11:02:08 2022

@author: Suklesh
"""
#Simple linear regression to predict salary
import pandas as pd
import numpy as np

data=pd.read_csv("F:\\ExcelR\\Assignments\\Simple Linear regression\\Salary_Data.csv")
data.shape
list(data)
data.corr()

#Exploratory Data Analysis
import seaborn as sns
X=data['YearsExperience']
X=X[:,np.newaxis]
X.ndim
Y=data['Salary']
sns.distplot(data['YearsExperience'])
data['YearsExperience'].hist() #Here as per the histogram, it looks like positively skewed
data['YearsExperience'].skew() #Skewness is 0.379, it can be accpeted as it is under range of -0.5 to +0.5
data['YearsExperience'].describe()

sns.distplot(data['Salary'])
data['Salary'].hist()
data['Salary'].skew()
data['Salary'].describe()

#Scatter plot
data.plot.scatter(x='YearsExperience',y='Salary')

#Fitting the model
from sklearn.linear_model import LinearRegression
model=LinearRegression().fit(X,Y)
model.intercept_
model.coef_
Y_pred=model.predict(X)

#To draw plots
import matplotlib.pyplot as plt
plt.scatter(X,Y,color='Blue')
plt.plot(X,Y_pred,color='Red')
plt.show()

#Finding error
Y_error=Y_pred-Y
sns.distplot(Y_error)

#Metrics
from sklearn.metrics import mean_squared_error
MSE=mean_squared_error(Y,Y_pred)
print(MSE)
from math import sqrt
RMSE=np.sqrt(MSE)
print(RMSE)

#################################################################################################

#Applying Transformations on X variable
#Exploratory Data Analysis
data['Cb YE']=np.sqrt(data['YearsExperience'])
X1=data['Cb YE']
X1=X1[:,np.newaxis]
sns.distplot(X1)
data['Cb YE'].skew() #Skewness is 0.379, it can be accpeted as it is under range of -0.5 to +0.5
data['Cb YE'].describe()

data['lg Salary']=np.log(data['Salary'])
Y1=data['lg Salary']
sns.distplot(Y1)
data['lg Salary'].skew()
data['lg Salary'].describe()

#Scatter plot
data.plot.scatter(x='Cb YE',y='lg Salary')

#Fitting the model
from sklearn.linear_model import LinearRegression
model1=LinearRegression().fit(X1,Y1)
model1.intercept_
model1.coef_
Y_pred1=model.predict(X1)

sns.regplot(x=X1,y=Y1,color='Blue')

import matplotlib.pyplot as plt
plt.scatter(X1,Y1,color='Blue')
plt.plot(X1,Y_pred1,color='Red')
plt.show()

#Finding error
Y_error1=Y1-Y_pred1
sns.distplot(Y_error1)

#Metrics
from sklearn.metrics import mean_squared_error
MSE1=mean_squared_error(Y1,Y_pred1)
print(MSE1)
from math import sqrt
RMSE1=np.sqrt(MSE1)
print(RMSE1)



#Splitting data into X and Y variables
X=data['S YE'] #X-->independent variable
X=X[:,np.newaxis] #Converting X from 1D to 2D 
X.ndim
data['YearsExperience'].hist()
data['S YE']=np.square(data['YearsExperience'])
data['SQ YE']=np.sqrt(data['YearsExperience'])
data['SQ YE'].hist()
data['SQ YE'].skew()
Y=data['Salary'] #Y--> dependent variable
data['Salary'].hist()

#Scatter plot
data.plot.scatter(x='YearsExperience',y='Salary')

#Fitting the model
from sklearn.linear_model import LinearRegression
model=LinearRegression().fit(X,Y)
model.intercept_
model.coef_
Y_pred=model.predict(X)

#To draw plots
import matplotlib.pyplot as plt
plt.scatter(X,Y,color='Blue')
plt.plot(X,Y_pred,color='Red')
plt.show()

#Finding error
Y_error=Y_pred-Y

#Metrics
from sklearn.metrics import mean_squared_error
MSE=mean_squared_error(Y,Y_pred)
print(MSE)
from math import sqrt
RMSE=np.sqrt(MSE)
print(RMSE)
