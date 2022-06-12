"""
Created on Fri Feb 25 11:21:09 2022
"""
#======================================================================
import pandas as pd
df = pd.read_csv("mushroom.csv")
df.shape
list(df)
df.head()

# finding missing values
df.isnull().sum()

#======================================================================

# lable encode
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

for eachcolumn in range(0,23):
    df.iloc[:,eachcolumn] = LE.fit_transform(df.iloc[:,eachcolumn])

df.head()
#======================================================================

# split as X and Y vairables
X = df.iloc[:,1:]
Y  = df['Typeofmushroom']

#======================================================================
# split your data in to two part - train and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, stratify=Y) # test_size=0.25

#======================================================================
# model development
from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()
MNB.fit(X_train,Y_train)

Y_pred = MNB.predict(X_test)


#======================================================================
# confusion matrix and accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(Y_test,Y_pred)
acc = accuracy_score(Y_test,Y_pred).round(2)

print("naive bayes model accuracy score:" , acc)
#======================================================================

X_test[0:3]
MNB.predict(X_test[0:3])

