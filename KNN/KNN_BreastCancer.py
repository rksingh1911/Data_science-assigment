"""
Created on Thu Mar 24 15:40:31 2022

"""
import pandas as pd

#pd.options.display.float_format = '{:.3f}'.format

df = pd.read_csv("breast-cancer-wisconsin-data.csv")
df.shape

df.head()

# split as X and Y
Y = df["diagnosis"]
X = df.iloc[:,2:]

# standardization
from sklearn.preprocessing import StandardScaler
X_scale = StandardScaler().fit_transform(X)
X_scale
type(X_scale)

'''
z = pd.DataFrame(X_scale)
z[0].describe()

z[10].describe()
z[10].hist()

'''
pd.crosstab(Y,Y)


###############################################################################
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X_scale, Y, stratify=Y ,random_state=42)  # By default test_size=0.25


# Install KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=1) # k =5 # p=2 --> Eucledian distance
knn.fit(X_train, y_train)

# Prediction
y_pred=knn.predict(X_test)

# Compute confusion matrix
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)

import numpy as np
print(np.mean(y_pred == y_test).round(3))  
print('Accuracy of KNN with K=5, on the test set: {:.3f}'.format(knn.score(X_test, y_test)))

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred).round(3)

knn.score(X_test, y_test).round(3)
# try with different k value and see where you get highest score
# that k value is final.

#====================================================
# Diabetic dataset
# take the data set of diabetic
'''
1. Apply logistic Regression and find accuracy score
2. Apply naive bayes  and find accuracy score
3. Apply KNN  and find accuracy score

and find the best algorithm which one gives higher accuracy

'''















