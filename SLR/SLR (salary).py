"""
Created on Mon Apr 11 01:55:35 2022

@author: ROHIT SINGH
"""
#Problem Statment Salary_hike -> Build a prediction model for Salary_hike

#Importing the libraries
import pandas as pd
import numpy as np

# Import .csv file and convert it to a DataFrame object
df = pd.read_csv("E:\\ExcelR\\assigment\\SLR\\Salary_Data (1).csv")
df.shape
df.head()
list(df)
type(df)
df.ndim
df.info()

# finding missing values
df.isnull().sum()

#Exploratory Data Analysis
import seaborn as sns
sns.distplot(df['YearsExperience'])
df['YearsExperience'].hist() #Here as per the histogram, it looks like positively skewed
df['YearsExperience'].skew() #Skewness is 0.379, it can be accpeted as it is under range of -0.5 to +0.5
df['YearsExperience'].describe()

sns.distplot(df['Salary'])
df['Salary'].hist()   #Here as per the histogram, it looks like positively skewed
df['Salary'].skew()   #Skewness is 0.35, it can be accpeted as it is under range of -0.5 to +0.5
df['Salary'].describe()

#pairPlot
sns.pairplot(df)

## Box and Whisker Plots
import matplotlib.pyplot as plt
plt.boxplot(df['YearsExperience'])
plt.boxplot(df['Salary'])

#Bar Graph plot
plt.bar(df['YearsExperience'], df['Salary'])
plt.xlabel("YearsExperience ")
plt.ylabel("Salary ")
plt.show()

#Pie Graph plot
plt.pie(df['YearsExperience'],                             
autopct ='% 1.1f %%', shadow = True)
plt.show()
    
#scatter plot
df.plot.scatter(x = 'YearsExperience',y = 'Salary')

#checking the co relation 
df.corr()

#normality Check
'''
# test of hypothesis  --> Normality test
Ho: Data is normal
H1: Data is not normal  
'''

from scipy.stats import shapiro
stat, p = shapiro(df['YearsExperience'])
print(stat)
print("p-value",p)

alpha = 0.05 # 5% level of significance

if p< alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted") #Data is normal


#transformation
x1 = np.sqrt(df['YearsExperience'])
#x1 = np.log(df['YearsExperience'])
sns.distplot(x1)
x1.skew() #Skewness is -0.0491, it can be accpeted as it is under range of -0.5 to +0.5
x1.describe()

#y1 = np.sqrt(df['Salary'])
y1 = np.log(df['Salary'])
sns.distplot(y1)
y1.skew() #Skewness is -0.0441, it can be accpeted as it is under range of -0.5 to +0.5
y1.describe()

# split as X and Y vairables
x1
x1.ndim
x = x1[:,np.newaxis] #changeing the dimention into 2D
x.ndim
y1
y = y1
y.ndim

#importing the linear regression and fitting the model
from sklearn.linear_model import LinearRegression
LE = LinearRegression()
model = LE.fit(x,y)
model.intercept_    ## To check the Bo values
model.coef_   ## To check the coefficients (B1)

# Makeing predictions using independent variable values
y_Pred = model.predict(x)

#plot output
import matplotlib.pyplot  as plt
plt.scatter(x,y, color = 'black')
plt.scatter(x,y_Pred,   color = 'yellow')
plt.plot(x, y_Pred, color = 'red')
plt.show()

#Finding Errors are the difference between observed and predicted values.
y_error = y-y_Pred
sns.distplot(y_error)
print(y_error)

#checking the result for best fit model like MSE and R_Square
from sklearn.metrics import mean_squared_error,r2_score
MSE =  mean_squared_error(y,y_Pred)
print(MSE.round(3))
Rmse = np.sqrt(MSE)
print(Rmse.round(3))
Rsquare = r2_score(y,y_Pred)
print(Rsquare.round(3))

#checking the summary for understaing
import statsmodels.api as sma
var1 = sma.OLS(y,x).fit()
var1.summary()




