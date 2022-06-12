"""
Created on Thu Apr  7 12:31:31 2022

@author: ROHIT SINGH
"""
import pandas as pd
bc = pd.read_csv("breast_cancer.csv")
bc
bc.shape
bc.head()
list(bc)
#working on lable encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
bc["Class_code"] = LE.fit_transform(bc["Class"])
bc[["Class","Class_code"]].head(11)
bc.shape
bc.head()
list(bc)
pd.crosstab(bc.Class, bc.Class_code)
# Split the data in to independent and Dependent
x = bc.iloc[:,1:10]
x.shape
x.head()
list(x)
y = bc.iloc[:,11]
y.shape
y.head()
list(y)
# Correlation check
# bc.corr()
#importing the logistic regassing module
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x,y)
logreg.intercept_
logreg.coef_
y_pred = logreg.predict(x)

from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score
cm = confusion_matrix(y,y_pred)
cm
print("accuracy score", (accuracy_score(y, y_pred)*100).round(3))
print("recall score",(recall_score(y,y_pred)*100).round(3))
print("precision score",(precision_score(y,y_pred)*100).round(3))
print("f1_score",(f1_score(y,y_pred)*100).round(3))

#manual calculation
TN = [0,0]
FN = [1,0]
FP = [0,1]
TP = [1,1]
specificity = TN /(TN+FP)
print("specificity" ,(specificity*100).round(3))

#_++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from sklearn.metrics import roc_curve,roc_auc_score
logreg.predict_proba(x).shape
logreg.predict_proba(x)[:,1]
print (logreg.predict_proba(x))
y_pred_proba = logreg.predict_proba(x)[:,1]
fpr, tpr,_ = roc_curve(y, y_pred_proba)

import matplotlib.pyplot as plt
plt.plot(fpr,tpr)
plt.ylabel('tpr - ture positive rate')
plt.xlabel('fpr - flase negative rate')
plt.show()
#auc score
auc = roc_auc_score(y,y_pred_proba)
auc


