"""
Created on Sun Jun 12 22:54:26 2022

@author: ROHIT SINGH
"""
#Business Problem
'''A cloth manufacturing company is interested to know about the segment or attributes causes high sale. 
Approach - A decision tree can be built with target variable Sale (we will first convert it in categorical variable) 
& all other variable will be independent in the analysis.  
'''

#Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#loading the data
df = pd.read_csv("E:\\ExcelR\\assigment\\DECISION TREE\\company_Data.csv")

#Exploratory Data Analysis
df.head()
df.tail()
df.shape
df.dtypes
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
outliar = sns.boxplot(df['Sales'])
# we have some outlier in in target variable 

#so here we have int and char both data we have to chenge it
#data cleaning 

"so all are continous variable till shelveloc so we can move it in last and then we can convert it "
x1=df['ShelveLoc']
df.drop(['ShelveLoc'],axis=1,inplace=True)
df_new=pd.concat([df,x1],axis=1)
df_new

#Converting Y variable to Categorival format
y1 = df_new['Sales']
y1_mean = y1.mean()
#creat empty set to store 
y2 = []

for i in range(0,400,1):
    if y1.iloc[i,]>=y1_mean:
        print('high')
        y2.append('high')
    else:
        print('Low')
        y2.append('Low')
y_new=pd.DataFrame(y2)
y1_new = y_new.set_axis(['Sales'],axis = 1)       
#saperating them to standarization and lable encoding
x1 = df_new.iloc[:,1:11]
x1.shape
list(x1.iloc[:,:7])

#standarization
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
mm = MinMaxScaler()
le = LabelEncoder()
x1.iloc[:,:7]=mm.fit_transform(x1.iloc[:,:7])
for i in range(7,10,1):
    x1.iloc[:,i]=le.fit_transform(x1.iloc[:,i])
print(x1)
x1.head()
list(x1)

#now we have to concate in the date fram
df1_new = pd.concat([x1,y1_new],axis =1)

# split as X and Y vairables
x = df1_new.iloc[:,0:10]
x.shape
x.dtypes
x.ndim

y = df1_new['Sales']
y.shape
y.dtypes
y.ndim

#Visualisation on data
x.hist(figsize= (20,7))
y.hist(figsize= (20,7))

#Displot
sns.distplot(df1_new['CompPrice'])
sns.distplot(df1_new['Income'])
sns.distplot(df1_new['Advertising'])
sns.distplot(df1_new['Population'])
sns.distplot(df1_new['Price'])
sns.distplot(df1_new['Age'])
sns.distplot(df1_new['Education'])
sns.distplot(df1_new['Urban'])
sns.distplot(df1_new['US'])
sns.distplot(df1_new['ShelveLoc'])
#we can obser the data is little bit normal shape

#count plot 
sns.countplot(df_new['Education'])
#we have equal no of  data

sns.countplot(df_new['Urban'])
#we have more yes no of urban data

sns.countplot(df_new['US'])
#we have more  no of us data

sns.countplot(df_new['Sales'])
#we have more good no as compare to risky data

#outlairs
df1_new.plot.box(figsize = (20,10))

#crosstab
pd.crosstab(df1_new.Age,df1_new.Sales).plot(kind = 'bar')
pd.crosstab(df1_new.Education,df1_new.Sales).plot(kind = 'bar')
pd.crosstab(df1_new.Urban,df1_new.Sales).plot(kind = 'bar')
#as we can see in above graph in urban area we have more sales

#density plot 
df1_new.plot(kind='density', subplots=True, layout=(4,5), figsize=(13,20), sharex=False, sharey=False)
plt.show()

#barplot
df1_new.plot(kind = 'bar',subplots=True, layout=(4,5), figsize=(13,20))

#line plot
df1_new.plot(kind = 'line',figsize = (13,20))

#pair plot
sns.pairplot(df1_new)

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

#Splitting the data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,stratify=y,random_state=74)
x_train.shape
y_train.shape
x_train.value_counts()
y_train.value_counts()

#Decision tree Classifier (As we have 2 outputs we choose Classifier)
from sklearn.tree import DecisionTreeClassifier
#DT = DecisionTreeClassifier(criterion='gini',max_depth=3)
#DT.fit(x_train,y_train)
DT1 = DecisionTreeClassifier(criterion='entropy',max_depth=8)
DT1.fit(x_train,y_train)

#make prediction
y_pred = DT1.predict(x_test)
y_pred.shape


#accuracy score
from sklearn.metrics import accuracy_score,confusion_matrix
ac = accuracy_score(y_test,y_pred)
print((ac*100).round(3))  #with gini 60% and entropy also same 84%
cm = confusion_matrix(y_test,y_pred)
print((cm*100).round(3))

#whose are right and wrogly predicted
df_predict=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
print(df_predict)

#tree ploting
from sklearn import tree
import graphviz 
Tree = tree.export_graphviz(DT1, out_file= None, 
                    filled=True, rounded=True,  
                    special_characters=True)  
graph = graphviz.Source(Tree)  
graph

#which column is more important
fi = pd.DataFrame({'feature': list(x_train.columns),
                   'importance': DT1.feature_importances_})
print(fi)
#we can see Income and Price ,CompPrice is most imp column
