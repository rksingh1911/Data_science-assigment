"""
Created on Sun Apr 24 17:52:27 2022
"""

#Import the libraries
import pandas as pd

# Import .csv file and convert it to a DataFrame object
df = pd.read_csv("D:\\CARRER\\My_Course\\ExcelR\\Data Science ExcelR\\Latest DS Material\\Day 18 - Kmeans , DBSCAN\\Wholesale customers data.csv")

print(df.head())


df

print(df.info())

df.drop(['Channel','Region'],axis=1,inplace=True)
array=df.values
array

from sklearn.preprocessing import StandardScaler
stscaler = StandardScaler().fit(array)
X = stscaler.transform(array)
X


from sklearn.cluster import DBSCAN
DBSCAN()
dbscan = DBSCAN(eps=3, min_samples=7)
dbscan.fit(X)

#Noisy samples are given the label -1.
dbscan.labels_

cl=pd.DataFrame(dbscan.labels_,columns=['cluster'])
cl
cl['cluster'].value_counts()


clustered = pd.concat([df,cl],axis=1)

noisedata = clustered[clustered['cluster']==-1]
finaldata = clustered[clustered['cluster']==0]

clustered

a=0
while a<5:
  print(a)
  a=a+1


clustered.mean()
finaldata.mean()

#by sir till this after mine-----------------------------------------------------------------------
for i in list(clustered):
    
    # show the list of values  
    print(clustered[i].tolist())
clustered.isnull().sum()

X = clustered.iloc[:,0:6]
Y  = clustered.iloc[:,6]


# split your data in to two part - train and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.20 ,random_state = 20, stratify=Y )

# model development
from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()
MNB.fit(X_train,Y_train)

Y_pred = MNB.predict(X_test)

# confusion matrix and accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(Y_test,Y_pred)
acc = accuracy_score(Y_test,Y_pred).round(2)

print("naive bayes model accuracy score:" , acc)