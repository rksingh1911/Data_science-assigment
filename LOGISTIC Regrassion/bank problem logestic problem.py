"""
Created on Mon Apr 11 00:02:01 2022

@author: ROHIT SINGH
"""

#Business Problem
'''Whether the client has subscribed a term deposit or not 
Binomial ("yes" or "no")'''

#Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#loading the data
df = pd.read_csv("E:\\ExcelR\\assigment\\LOGISTIC Regrassion\\bank-full.csv",sep = ';')

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

#saperating continuous and categorical data
x1 = df[['age','balance','duration','campaign','previous','pdays']]
x2 = df[['job','marital','education','month','housing','loan','y']]
 
     
#Visualisation on data
      
#histogram
x1.hist(figsize=(20,7))

#countplot
sns.countplot(x1['age'])
sns.countplot(x1['day'])
sns.countplot(x1['balance'])
sns.countplot(x1['duration'])
sns.countplot(x1['campaign'])
sns.countplot(x1['previous'])
sns.countplot(x1['pdays'])

#pairplot
sns.pairplot(df)

#Stackedbarchart&unstackedbarchart
df.plot.bar(figsize=(15, 8))
df.plot.bar(stacked = True,figsize=(15, 8))

#Percentage of Client Subscribed
df['y'].value_counts()
count_no_sub = len(df[df['y']=="no"])
count_yes_sub = len(df[df['y']=="yes"])
count_yes_sub/(count_no_sub+count_yes_sub)*100
"Percentage of Client Subscribed is 11.70 % in the current data set"

#Subscribed whose age are
pd.crosstab(df.age,df.y).plot(kind = 'line')
"Most of the customers are in age between 20 and 60 years"

#Subscribed for duration
pd.crosstab(df.duration,df.y).plot(kind = 'line')
"Most of the customers are in duration for between 20 to 800"

#Subscribed Frequency for Job Title
pd.crosstab(df.job,df.y).plot(kind='bar')
"The frequency of subscribtion depends a great deal on the job title. Thus, the job title can be a good predictor of the outcome variable"

#Stacked Bar Chart of Marital Status vs Subscribed
pd.crosstab(df.marital,df.y).plot(kind = 'bar',stacked = True)
df['marital'].value_counts()
"The marital status seem a strong predictor for the target variable"

#Stacked Bar Chart of Education vs Subscribed
pd.crosstab(df.education,df.y).plot(kind = 'bar',stacked = True)
df['education'].value_counts()
"Stacked Bar Chart of Education vs Subscribed"

#Bar Chart of Contact vs Subscribed
pd.crosstab(df.month,df.y).plot(kind = 'bar')
df['month'].value_counts()
"in the month of may and june people subscribed most"

#Stacked Bar Chart of housing vs Subscribed
pd.crosstab(df.housing,df.y).plot(kind = 'bar',stacked = True)
df['housing'].value_counts()
"Data is somewhat evenly distributed on whether the client has House or not"

#Bar Chart of loan vs Subscribed
pd.crosstab(df.loan,df.y).plot(kind='bar')
df['loan'].value_counts()
"majority of the client do not have loan"

#Stacked Bar Chart of Contact vs Subscribed
pd.crosstab(df.contact,df.y).plot(kind = 'bar')
"Contact does not seem a strong predictor for the outcome variable"

#DATA cleaning
df.isnull().sum()  #Since there are no Null values in any column

#normality
#Test of hypothesis 
from scipy.stats import shapiro
stat, p = shapiro(x1)
print(stat)
print("p-value",p)

alpha = 0.05 # 5% level of significance

if p < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")

# H1: Data is normal

#Chi_square_test for categorical
crosstab = x2.value_counts()
import scipy.stats as stats
result = stats.chi2_contingency(crosstab)
print(result)
#here we got Z value and P value and array
P = 0.0
alpha = 0.05

if P < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")
# H1: Data is normal    

#Logistic Regression Model

# standardization of dependent variables
from sklearn import preprocessing
x3 = preprocessing.scale(x1)
df1 = pd.DataFrame(x3)
df2 = df1.set_axis([['age','balance','duration','campaign','previous','pdays']],axis=1)

#lableEncoding for categorical data
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
for i in range(0,7,1):
    x2.iloc[:,i]=LE.fit_transform(x2.iloc[:,i])
print(x2)
df3 = pd.DataFrame(x2)

#concating the both data set into one data set 
df4 = pd.concat([df2,df3],axis = 1)
df4
list(df4)
# split as X and Y vairables
x = df4.iloc[:,:13]

y = df4['y']

#chechking dimention
x.ndim

# Import Logistic Regression
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(x,y)
LR.intercept_
LR.coef_

# Make predictions using independent variable values
Y_Pred = LR.predict(x)

#checking the result for best fit model like accuracy score and sensitivity
from sklearn.metrics import confusion_matrix,recall_score,accuracy_score,precision_score,f1_score

cm  = confusion_matrix(y,Y_Pred)
print(cm)
TN = cm[0,0]
FN = cm[1,0]
FP = cm[0,1]
TP = cm[1,1]

print("Accuracy_score:",(accuracy_score(y,Y_Pred)*100).round(3))
print("recall/senstivity_score",(recall_score(y,Y_Pred)*100).round(3))
print("precision_score",(precision_score(y,Y_Pred)*100).round(3))
print("F1_score",(f1_score(y,Y_Pred)*100).round(3))
specificity  = TN/(TN+FP)
print("specificity",(specificity*100).round(3))

#checking classification report
from sklearn.metrics import classification_report
print(classification_report(y, Y_Pred))

#checking the summary for understaing
import statsmodels.api as sma
log_reg = sma.Logit(y,x).fit()
print(log_reg.summary())

#Output Interpretation
'''Confusion Matrix
The result is telling us that we have 39455+456 correct predictions and 4833+467 incorrect predictions
Accuracy == 89%
Of the entire data set, 84% of the clients will subcribe'''