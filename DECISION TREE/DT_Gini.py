# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 12:01:11 2022

@author: Hi
"""

import pandas as pd
df = pd.read_csv("Cricket.csv")
df.head()
list(df)

# preprocesssing
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df["Gender"] = LE.fit_transform(df["Gender"])
df["Class"] = LE.fit_transform(df["Class"])
df["Cricket"] = LE.fit_transform(df["Cricket"])

X =  df.iloc[:,:2]
Y = df["Cricket"]



#fit the model
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='gini')
DT.fit(X,Y)

Y_pred = DT.predict(X)

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y,Y_pred)
cm
ac = accuracy_score(Y,Y_pred)
ac

#----------------------------------
import numpy as np
X_t = np.array([[0,1]])
DT.predict(X_t)
#----------------------------------

##############################################################################
from sklearn import tree
import graphviz 
dot_data = tree.export_graphviz(DT, out_file= None, 
                    filled=True, rounded=True,  
                    special_characters=True)  
graph = graphviz.Source(dot_data)  
graph