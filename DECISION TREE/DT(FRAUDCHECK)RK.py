"""
Created on Sun Jun 12 22:57:46 2022

@author: ROHIT SINGH
"""
#Business Problem
'''Use decision trees to prepare a model on fraud data 
treating those who have taxable_income <= 30000 as "Risky" 
and others are "Good"
'''
#Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#loading the data
df = pd.read_csv("E:\\ExcelR\\assigment\\DECISION TREE\\Fraud_check.csv")

#Exploratory Data Analysis
df.head()
df.tail()
df.shape
list(df)
type(df)
df.ndim
df.info()

#describe for all mean mode information
df.describe()

#finding null value
df.isnull()
df.isnull().sum()

#skew for normal distributaion
df.skew()  #so  all variable skewness are normal 

#corelation
df.corr()

#Outlier Check
outliar = sns.boxplot(df['Taxable.Income'])
#we have no outliar

#so here we have int and char both data we have to chenge it
#data cleaning 

y1= df['Taxable.Income']
y1.shape
df.drop(['Taxable.Income'],axis = 1, inplace = True)
df
#so here frist we have to mapp the data point with <=30000 as risky and >=30000 as good 
#creat empty set to store 
y2 = []

for i in range(0,600,1):
    if y1.iloc[i,]<=30000:
        print('Risky')
        y2.append('Risky')
    else:
        print('Good')
        y2.append('Good')

print(y2)
#now we have to concate in the date fram and do lable encodeing 
y3 = pd.DataFrame(y2)
y3.set_axis(['Tax'],axis = 1,inplace = True)
df_new=pd.concat([df,y3],axis=1)
df_new


# split as X and Y vairables
x = df_new.iloc[:,0:5]
x.shape
x.dtypes
x.ndim

y = df_new['Tax']
y.shape
y.dtypes
y.ndim
#lable encoding
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
x['Undergrad']=LE.fit_transform(x['Undergrad'])
x['Marital.Status']=LE.fit_transform(x['Marital.Status'])
x['Urban']=LE.fit_transform(x['Urban'])

#minmax Scaler
from sklearn.preprocessing import MinMaxScaler
me = MinMaxScaler()
x[['City.Population','Work.Experience']] = me.fit_transform(x[['City.Population','Work.Experience']])

#Visualisation on data

#Histogram
x.hist(figsize= (20,7))
y.hist(figsize= (20,7))

#Displot 
sns.distplot(df_new['City.Population'])
sns.distplot(df_new['Work.Experience'])
#we can obser the data is little bit normal shape 

#count plot 
sns.countplot(df_new['Undergrad'])
#we have equal no of undegrad data

sns.countplot(df_new['Marital.Status'])
#we have equal no of mrrital status data

sns.countplot(df_new['Urban'])
#we have equal no of urban data

sns.countplot(df_new['Tax'])
#we have more good no as compare to risky data

#crosstab
pd.crosstab(df_new.Undergrad,df_new.Tax).plot(kind = 'bar')
pd.crosstab(df_new.Urban,df_new.Tax).plot(kind = 'bar')
# so in both graph we can see urban area pepole and undergra people they both are equaly good and risky

#density plot 
df_new.plot(kind='density', subplots=True, layout=(4,5), figsize=(13,20), sharex=False, sharey=False)
plt.show()

#barplot
df_new.plot(kind = 'bar',subplots=True, layout=(4,5), figsize=(13,20))

#line plot
df_new.plot(kind = 'line',figsize = (13,20))

#pair plot
sns.pairplot(df_new)

#normality test

#Test of hypothesis 
from scipy.stats import shapiro
stat, p = shapiro(x)
print(stat)
print("p-value",p)

alpha = 0.05 # 5% level of significance

if p < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")

# H1: Data is normal

#Decision tree

#Splitting the data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,stratify=y,random_state=99)
x_train.shape
y_train.shape
x_train.value_counts()
y_train.value_counts()

#Decision tree Classifier (As we have 2 outputs we choose Classifier)
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='gini',max_depth=3)
DT.fit(x_train,y_train)
#DT1 = DecisionTreeClassifier(criterion='entropy',max_depth=3)
#DT1.fit(x_train,y_train)

#make prediction
y_pred = DT.predict(x_test)
y_pred.shape


#accuracy score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
ac = accuracy_score(y_test,y_pred)
print((ac*100).round(3))  #with gini 80% and entropy also same 80%
cm = confusion_matrix(y_test,y_pred)
print((cm*100).round(3))
cl_report = classification_report(y_test,y_pred)
print(cl_report)
#whose are right and wrogly predicted
df_predict=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
print(df_predict)

#tree ploting
from sklearn import tree
import graphviz 
Tree = tree.export_graphviz(DT, out_file= None, 
                    filled=True, rounded=True,  
                    special_characters=True)  
graph = graphviz.Source(Tree)  
graph

#which column is more important
fi = pd.DataFrame({'feature': list(x_train.columns),
                   'importance': DT.feature_importances_})
print(fi)
#we can see city.population and work.experience is most imp column























