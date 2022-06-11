"""
Created on Mon Apr 25 01:04:52 2022

@author: ROHIT SINGH
"""
#Problem Statment Delivery_time -> Predict delivery time using sorting time#

#Importing the libraries
import pandas as pd
import numpy as np

# Import .csv file and convert it to a DataFrame object
df = pd.read_csv("E:\\ExcelR\\assigment\\SLR\\delivery_time.csv")
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
sns.distplot(df['Sorting Time'])
df['Sorting Time'].hist() #Here as per the histogram, it looks like positively skewed
df['Sorting Time'].skew() #Skewness is 0.047, it can be accpeted as it is under range of -0.5 to +0.5
df['Sorting Time'].describe()

sns.distplot(df['Delivery Time'])
df['Delivery Time'].hist()   #Here as per the histogram, it looks like positively skewed
df['Delivery Time'].skew()   #Skewness is 0.35, it can be accpeted as it is under range of -0.5 to +0.5
df['Delivery Time'].describe()

#pairPlot
sns.pairplot(df)

## Box and Whisker Plots
import matplotlib.pyplot as plt
plt.boxplot(df['Sorting Time'])
plt.boxplot(df['Delivery Time'])

#Bar Graph plot
plt.bar(df['Sorting Time'], df['Delivery Time'])
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()

#Pie Graph plot
plt.pie(df['Sorting Time'],                             
autopct ='% 1.1f %%', shadow = True)
plt.show()
  
# scatter plot
df.plot.scatter(x='Sorting Time', y='Delivery Time')

#checking the co relation
df.corr()

#normality Check
'''
# test of hypothesis  --> Normality test
Ho: Data is normal
H1: Data is not normal  
'''

from scipy.stats import shapiro
stat, p = shapiro(df['Sorting Time'])
print(stat)
print("p-value",p)

alpha = 0.05 # 5% level of significance

if p< alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted") #Data is normal

#transformation
#x1 = np.sqrt(df['Sorting Time'])
x1 = np.log(df['Sorting Time'])
x2 = np.log(x1)
sns.distplot(x1)
x1.skew() #Skewness is -0.25, it can be accpeted as it is under range of -0.5 to +0.5
x1.describe()

#y1 = np.sqrt(df['Delivery Time'])
y1= np.log(df['Delivery Time'])
sns.distplot(y1)
y1.skew() #Skewness is -0.45, it can be accpeted as it is under range of -0.5 to +0.5
y1.describe()

# split as X and Y vairables
x1
x1.ndim
x = x1[:,np.newaxis] #changeing the dimention into 2D
x.ndim
y1
y = y1
y.ndim

# Import Linear Regression and fitting the model
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(x, y)
model.intercept_  ## To check the Bo values
model.coef_       ## To check the coefficients (B1)

# Makeing predictions using independent variable values
y_Pred = model.predict(x)

#plot output
import matplotlib.pyplot  as plt
plt.scatter(x,y, color = 'black')
plt.scatter(x,y_Pred,   color = 'yellow')
plt.plot(x, y_Pred, color = 'red')
plt.show()

# Errors are the difference between observed and predicted values.
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
