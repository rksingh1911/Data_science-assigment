"""
Created on Thu Jun  2 18:27:43 2022

@author: ROHIT SINGH
"""
#Business Problem
'''
Consider only the below columns and prepare a prediction model 
for predicting Price_Corolla<-Corolla
[c("Price","Age_08_04","KM","HP","cc","Doors","Gears","QuarterlyTax","Weight")]
'''

#Importing the libraries
import numpy as np
import pandas as pd

#loading the data
df = pd.read_csv("E:\\ExcelR\\assigment\\MLR\\ToyotaCorolla.csv",encoding='latin1')
df.shape
df.head()
list(df)
type(df)
df.ndim
df.info()

'''here we can see in the variable which are not important so as now
we need to drop those variable from out data set'''

#DATA CLEANING
df.drop(['Id','Model','Mfg_Month','Mfg_Year','Fuel_Type','Met_Color','Color','Automatic','Cylinders','Mfr_Guarantee',
        'BOVAG_Guarantee','Guarantee_Period','ABS','Airbag_1','Airbag_2','Airco','Automatic_airco','Boardcomputer',
        'CD_Player','Central_Lock','Powered_Windows', 'Power_Steering','Radio','Mistlamps','Sport_Model','Backseat_Divider',
         'Metallic_Rim', 'Radio_cassette', 'Tow_Bar'],axis = 1,inplace = True)

df

# finding missing values
df.isnull().sum()

#Exploratory Data Analysis

#univariate plots

#histogram
df.hist(figsize=(20,7))

import seaborn as sns
#KDEplot
sns.kdeplot(df['Price'])
sns.kdeplot(df['KM'])
sns.kdeplot(df['Age_08_04'])
sns.kdeplot(df['HP'])
sns.kdeplot(df['cc'])
sns.kdeplot(df['Doors'])
sns.kdeplot(df['Gears'])
sns.kdeplot(df['Quarterly_Tax'])
sns.kdeplot(df['Weight'])
            
#boxplot
sns.boxplot(df['Price'])
sns.boxplot(df['KM'])
sns.boxplot(df['Age_08_04'])
sns.boxplot(df['HP'])
sns.boxplot(df['cc'])
sns.boxplot(df['Doors'])
sns.boxplot(df['Gears'])
sns.boxplot(df['Quarterly_Tax'])
sns.boxplot(df['Weight'])
#we can observe there are too much outliar so that can effect our result

#RUGPLOT
sns.rugplot(df['Price'])
sns.rugplot(df['KM'])
sns.rugplot(df['Age_08_04'])
sns.rugplot(df['HP'])
sns.rugplot(df['cc'])
sns.rugplot(df['Doors'])
sns.rugplot(df['Gears'])
sns.rugplot(df['Quarterly_Tax'])
sns.rugplot(df['Weight'])

#violinplot
sns.violinplot(df['Price'])
sns.violinplot(df['KM'])
sns.violinplot(df['Age_08_04'])
sns.violinplot(df['HP'])
sns.violinplot(df['cc'])
sns.violinplot(df['Doors'])
sns.violinplot(df['Gears'])
sns.violinplot(df['Quarterly_Tax'])
sns.violinplot(df['Weight'])

#stripplot
sns.stripplot(df['Price'])
sns.stripplot(df['KM'])
sns.stripplot(df['Age_08_04'])
sns.stripplot(df['HP'])
sns.stripplot(df['cc'])
sns.stripplot(df['Doors'])
sns.stripplot(df['Gears'])
sns.stripplot(df['Quarterly_Tax'])
sns.stripplot(df['Weight'])

#countplot
sns.countplot(df['Price'])
sns.countplot(df['KM'])
sns.countplot(df['Age_08_04'])
sns.countplot(df['HP'])
sns.countplot(df['cc'])
sns.countplot(df['Doors'])
sns.countplot(df['Gears'])
sns.countplot(df['Quarterly_Tax'])
sns.countplot(df['Weight'])

#distplot
sns.distplot(df['Price'])
sns.distplot(df['KM'])
sns.distplot(df['Age_08_04'])
sns.distplot(df['HP'])
sns.distplot(df['cc'])
sns.distplot(df['Doors'])
sns.distplot(df['Gears'])
sns.distplot(df['Quarterly_Tax'])
sns.distplot(df['Weight'])

#BIvariateplot

#Scatterplot
df.plot.scatter(x='KM',y='Price')
df.plot.scatter(x='Age_08_04',y='Price')
df.plot.scatter(x='HP',y='Price')
df.plot.scatter(x='cc',y='Price')
df.plot.scatter(x='Doors',y='Price')
df.plot.scatter(x='Gears',y='Price')
df.plot.scatter(x='Quarterly_Tax',y='Price')
df.plot.scatter(x='Weight',y='Price')

#bivariatelinechart
df.plot.line(x='KM',y='Price')
df.plot.line(x='Age_08_04',y='Price')
df.plot.line(x='HP',y='Price')
df.plot.line(x='cc',y='Price')
df.plot.line(x='Doors',y='Price')
df.plot.line(x='Gears',y='Price')
df.plot.line(x='Quarterly_Tax',y='Price')
df.plot.line(x='Weight',y='Price')
#Stackedbarchart&unstackedbarchart
df.plot.bar(figsize=(15, 8))
df.plot.bar(stacked = True,figsize=(15, 8))
#BARgraph
sns.barplot(x='KM',y='Price',data = df,color='red',)
sns.barplot(x = 'Age_08_04',y = 'Price',data = df,color='green',)
sns.barplot(x = 'HP',y = 'Price',data = df,color='yellow',)
sns.barplot(x='cc',y='Price',data = df,color='red',)
sns.barplot(x='Doors',y='Price',data = df,color='red',)
sns.barplot(x='Gears',y='Price',data = df,color='red',)
sns.barplot(x='Quarterly_Tax',y='Price',data = df,color='red',)
sns.barplot(x='Weight',y='Price',data = df,color='red',)

#pairplot
sns.pairplot(df)
#skew for normal distributaion
df.skew()  #for all variable skewness in under -0.5 to +0.5
#describe for all mean mode information
df.describe()
#corelation
df.corr()

#normality
#Test of hypothesis 
from scipy.stats import shapiro
stat, p = shapiro(df)
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
X = df[['Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight']]

# standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scale = scaler.fit_transform(X)
x_new = pd.DataFrame(X_scale)
x =  x_new.set_axis([["Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]],axis=1)
type(x)

# split as X and Y vairables

#x = df[['Age_08_04','KM','HP','Gears','Quarterly_Tax','Weight']]
#x = df[['Age_08_04','KM','HP','cc','Doors','Quarterly_Tax','Weight']]  #P val Doors = 0.968>0.005
#x = df[['Age_08_04','KM','HP','cc','Quarterly_Tax','Weight']] #P val cc = 0.169>0.005 
#x = df[['Age_08_04','KM','HP','Quarterly_Tax','Weight']] #P val 
x = np.sqrt(df[['Age_08_04','KM','HP','Quarterly_Tax','Weight']])  #PERFECTLY FIT
#x = np.sqrt(df[['Age_08_04','KM','HP','cc','Quarterly_Tax','Weight']])
y = df['Price']

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

'''As expected, gears, doors,cc,HP, has the lowest variance inflation 
factor We would need to discard this variables to improve 
model and try to solve multicolinearity.'''

#checking the summary for understaing
import statsmodels.api as sma
var1 = sma.OLS(y,x).fit()
var1.summary()

'''
#result  #R_squred  warning
frist  = 83.  multicollinearty
second = 98   multicollinearty
trird  = 98.6 multicollinearty
fouth  = 98.7 multicollinearty
fifth = 98.7  no multicollinearty
sixth = 98.7 no multicollinearty
'''