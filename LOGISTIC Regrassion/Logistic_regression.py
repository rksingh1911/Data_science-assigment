"""
Created on Thu Apr  7 11:15:34 2022
"""

"""
LOGISTIC REGRESSION USING BREAST CANCER DATASET
"""
# ignore all future warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

#import pandas
import pandas as pd
bc = pd.read_csv("breast_cancer.csv")
bc.shape


bc.head()
list(bc)

# working on label encoding
from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()
bc["Class_code"] = LE.fit_transform(bc["Class"])
bc[["Class", "Class_code"]].head(11)
bc.shape
bc.head()
list(bc)

pd.crosstab(bc.Class,bc.Class_code)

# Split the data in to independent and Dependent
X = bc.iloc[:,1:10] # Only Independent variables
X.shape
list(X)
Y = bc.iloc[:,11]  # Only Dependent variable Sales
Y.shape
list(Y)

# Correlation check
# bc.corr()


# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X,Y)

# 
logreg.intercept_  ## To check the Bo values
logreg.coef_       ## To check the coefficients (B1,B2,...B8)

#
Y_Pred=logreg.predict(X)

#==============================================================================
# Compute confusion matrix

# comparision
from sklearn.metrics import confusion_matrix, accuracy_score,recall_score,precision_score, f1_score
CM = confusion_matrix(Y, Y_Pred)
CM

# Manual calculations
TN = CM[0,0]
FN = CM[1,0]
FP = CM[0,1]
TP = CM[1,1]

# sklearn calculations
print("Accuracy_score:",(accuracy_score(Y,Y_Pred)*100).round(3))
print("Recall/Sensitivity score:",(recall_score(Y,Y_Pred)*100).round(3))
print("Precision score:",(precision_score(Y,Y_Pred)*100).round(3))

Specificity = TN /(TN + FP)
print("Specificity score: ",(Specificity*100).round(3))
print("F1 score: ",(f1_score(Y,Y_Pred)*100).round(3))


#==============================================================================

# Show confusion matrix in a separate window
import matplotlib.pyplot as plt
plt.matshow(CM)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#==============================================================================

