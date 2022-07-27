"""
Created on Thu Jun 16 21:48:25 2022

@author: ROHIT SINGH
"""

#Business Problem
'''Use Random Forest to prepare a model on fraud data 
treating those who have taxable_income <= 30000 as "Risky" and 
others are "Good"
'''

#Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#loading the data
df = pd.read_csv("E:\\ExcelR\\assigment\\RANDOM FOREST\\Fraud_check (1).csv")

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
#we dont have outlier 

#so here we have int and char both data we have to chenge it
#data cleaning

x1=df[['Undergrad','Marital.Status','Urban']]
x2=df[['City.Population','Work.Experience']]

#Data preprocessing
from sklearn.preprocessing import LabelEncoder,StandardScaler
le=LabelEncoder()
sc=StandardScaler()

for i in range(0,3,1):
    x1.iloc[:,i]=le.fit_transform(x1.iloc[:,i])
    
x_scale=sc.fit_transform(x2)
x_scale=pd.DataFrame(x_scale)
x_scale.set_axis(['City.Population','Work.Experience'],axis='columns',inplace=True)

'''As per the problem statement we are asked to convert this Y variable into categorical
So, Differentiating the Y variable with respect to 30000 (as mentioned in the prob statement)'''

y1= df['Taxable.Income']
y1.shape
df.drop(['Taxable.Income'],axis = 1, inplace = True)
df

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
df_new=pd.concat([x1,x_scale,y3],axis=1)
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

#RANDOM FOREST

#Splitting the data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,stratify=y,random_state=67)
x_train.shape
y_train.shape
x_train.value_counts()
y_train.value_counts()

#Random Forest Classifier(As we have 2 outputs we choose Classifier)
from sklearn.ensemble import RandomForestClassifier
RFC=RandomForestClassifier(max_features=0.5,n_estimators=500)
RFC.fit(x_train,y_train)

#make prediction
y_pred = RFC.predict(x_test)
y_pred.shape

#accuracy score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc = accuracy_score(y_test,y_pred)
print((acc*100).round(3)) 
cm = confusion_matrix(y_test,y_pred)
print((cm*100).round(3))
cl_report = classification_report(y_test,y_pred)
print(cl_report)

# Loop with the  DecisionTreeRegressor through the Max_feature values to determine the most appropriate (best)
acc1=[]
set1=np.arange(0.1,1.1,0.1)
for i in set1:
    RFC=RandomForestClassifier(max_features=i,n_estimators=500)
    RFC.fit(x_train,y_train)
    y_pred=RFC.predict(x_test)
    acc=accuracy_score(y_test, y_pred)
    acc1.append((acc*100).round(3))
    print('For max features',i,',accuracy is',(acc*100).round(3))

#plot the graph between maxfeature and acc    
import matplotlib.pyplot as plt
plt.plot(set1,acc1,data=None)
plt.xlabel('max_features')
plt.ylabel('accuracy')
plt.title('Graph between max features and accuracy')
plt.show()
    
#whose are right and wrogly predicted
df_predict=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
print(df_predict)

#which column is more important
fi = pd.DataFrame({'feature': list(x_train.columns),
                   'importance': RFC.feature_importances_})
print(fi) 
#citipopulation  and work experience is most important variable 
  
#max feature with 0.5 we got 79% of accuracy   
    
