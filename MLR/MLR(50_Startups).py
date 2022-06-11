"""
Created on Tue Apr 26 01:01:49 2022

@author: ROHIT SINGH
"""

#Bussiness Problem
'''Prepare a prediction model for profit of 50startups data
Do transformations for getting better predictions of profit and
make a table containing R^2 value for each prepared model'''


#Importing the libraries
import numpy as np
import pandas as pd

#loading the data
df = pd.read_csv("E:\\ExcelR\\assigment\\MLR\\50_Startups.csv")
df.shape
df.head()
list(df)
type(df)
df.ndim
df.info()

'''here we can see in the variable state is catagorical so as foe now we leave that
column or we can do lable encoding if it is really effecting on model '''


# finding missing values
df.isnull().sum()

#Exploratory Data Analysis

#univariate plots

#histogram
df.hist(figsize=(20,7))

import seaborn as sns
#KDEplot
sns.kdeplot(df['R&D Spend'])
sns.kdeplot(df['Administration'])
sns.kdeplot(df['Marketing Spend'])
sns.kdeplot(df['Profit'])
#boxplot
sns.boxplot(df['R&D Spend'])
sns.boxplot(df['Administration'])
sns.boxplot(df['Marketing Spend'])
sns.boxplot(df['Profit'])
#RUGPLOT
sns.rugplot(df['R&D Spend'])
sns.rugplot(df['Administration'])
sns.rugplot(df['Marketing Spend'])
sns.rugplot(df['Profit'])
#violinplot
sns.violinplot(df['R&D Spend'])
sns.violinplot(df['Administration'])
sns.violinplot(df['Marketing Spend'])
sns.violinplot(df['Profit'])
#stripplot
sns.stripplot(df['R&D Spend'])
sns.stripplot(df['Administration'])
sns.stripplot(df['Marketing Spend'])
sns.stripplot(df['Profit'])
#countplot
sns.countplot(df['R&D Spend'])
sns.countplot(df['Administration'])
sns.countplot(df['Marketing Spend'])
sns.countplot(df['Profit'])
#distplot
sns.distplot(df['R&D Spend'])
sns.distplot(df['Administration'])
sns.distplot(df['Marketing Spend'])
sns.distplot(df['Profit'])
#BIvariateplot

#Scatterplot
df.plot.scatter(x='R&D Spend',y='Profit')
df.plot.scatter(x='Administration',y='Profit')
df.plot.scatter(x='Marketing Spend',y='Profit')
#Hexplot
df.plot.hexbin(x='R&D Spend',y='Profit')
df.plot.hexbin(x='Administration',y='Profit')
df.plot.hexbin(x='Marketing Spend',y='Profit')
#bivariatelinechart
df.plot.line(x='R&D Spend',y='Profit')
df.plot.line(x='Administration',y='Profit')
df.plot.line(x='Marketing Spend',y='Profit')
#Stackedbarchart&unstackedbarchart
df.plot.bar(figsize=(15, 8))
df.plot.bar(stacked = True,figsize=(15, 8))
#BARgraph
sns.barplot(x = 'Administration',y = 'Profit',data = df,color='red',)
sns.barplot(x = 'R&D Spend',y = 'Profit',data = df,color='green',)
sns.barplot(x = 'Marketing Spend',y = 'Profit',data = df,color='yellow',)
#piechart
from matplotlib import pyplot as plt
size = df['Profit']
marketing = df['Marketing Spend']
spend = df['R&D Spend']
Administration = df['Administration']
fig = plt.figure(figsize = (10, 5))
plt.pie(size,labels =marketing,autopct='%1.1f%%',shadow = True)
fig = plt.figure(figsize = (10, 5))
plt.pie(size,labels =spend,autopct='%1.1f%%',shadow = True)
fig = plt.figure(figsize = (10, 5))
plt.pie(size,labels =Administration,autopct='%1.1f%%',shadow = True)
#pairplot
sns.pairplot(df)
#skew for normal distributaion
df.skew()  #for all variable skewness in under -0.5 to +0.5
#describe for all mean mode information
df.describe()
#corelation
df.corr()

#DATA CLEANING

df_new = df.drop(['State'],axis = 1) #we have drop State variable 

#normality
#Test of hypothesis 
from scipy.stats import shapiro
stat, p = shapiro(df_new)
print(stat)
print("p-value",p)

alpha = 0.05 # 5% level of significance

if p < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")

# H1: Data is normal

#MODEL FITTING

# Assumptions for the Linear Regression
# Linear relationship should exists between X and Y
# All variables and its samples should be drawn from Normal distribution


# defineing the dependent variables
X = df_new[['R&D Spend','Administration','Marketing Spend']]


# standardization of dependent variables
from sklearn import preprocessing
data_new = preprocessing.scale(X)
x_new = pd.DataFrame(data_new)
x = x_new.set_axis([['R&D Spend','Administration','Marketing Spend']],axis=1)
type(x)

# split as X and Y vairables
# x = df_new[['R&D Spend','Administration','Marketing Spend']]
# x = df_new[['R&D Spend','Marketing Spend']]
# x = np.sqrt(df_new[['R&D Spend','Administration','Marketing Spend']])
x = np.sqrt(df_new[['R&D Spend','Marketing Spend']])
y = df_new['Profit']

#chechking dimention
x.ndim

# Import Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(x,y)
model.intercept_  ## To check the Bo values
model.coef_       ## To check the coefficients (B1)

# Make predictions using independent variable values
Y_Pred = model.predict(x)

#plot output
import matplotlib.pyplot as plt
plt.scatter(y,Y_Pred);
plt.xlabel('Actual');
plt.ylabel('Predicted');
sns.regplot(x=y,y=Y_Pred,ci=None,color='red');

# Errors are the difference between observed and predicted values.
y_error = y-Y_Pred
sns.distplot(y_error)
print(y_error)

#checking the result for best fit model like MSE and R_Square
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y,Y_Pred)
print("Mean square error: ", (mse).round(3))

from sklearn.metrics import r2_score
r2 = r2_score(y,Y_Pred)*100
print("R square: ", r2.round(3))

#Finding Variance Inflation Factor (VIF)
import statsmodels.api as sm
X1 = sm.add_constant(x) ## let's add an intercept (beta_0) to our model
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(X1.values, j) for j in range(X1.shape[1])]
variable_VIF = pd.concat([pd.DataFrame(X1.columns),pd.DataFrame(np.transpose(vif))], axis = 1)
print(variable_VIF)

'''As expected, Administration has the lowest variance inflation 
factor We would need to discard this variables to improve 
model and try to solve multicolinearity.'''

#checking the summary for understaing
import statsmodels.api as sma
var1 = sma.OLS(y,x).fit()
var1.summary()

'''
#result  #R_squred 
frist  = 98.7
second = 95.6
trird  = 98.8
fouth  = 98.7
'''
#so in this four result we ll consider fourth one because with 'R&D Spend','Marketing Spend giving better R2squerd result
