
###############################################################################
import pandas as pd  
#import numpy as np  

customer_data = pd.read_csv('shopping_data.csv', delimiter=',') 
customer_data.shape
customer_data.head()
X = customer_data.iloc[:, 2:5].values 
X.shape
##############################################################################
#matplotlib qt
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2])
plt.show()

# Initializing KMeans
from sklearn.cluster import KMeans
KMeans()
kmeans = KMeans(n_clusters=5)
# Fitting with inputs
kmeans = kmeans.fit(X)
# Predicting the clusters
labels = kmeans.predict(X)
type(labels)
# Getting the cluster centers
C = kmeans.cluster_centers_
# Total with in centroid sum of squares 
kmeans.inertia_


#matplotlib qt
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2])
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='Red', s=1000) # S is star size, c= * color
#########################################################
"""           
ltype = pd.DataFrame(labels)

customer_data['Is_ltype'] = pd.Categorical(ltype)
print (customer_data.Is_ltype)



pd.crosstab(customer_data['Genre'],customer_data['Genre'])



type(ltype)
"""

Y = pd.DataFrame(labels)
X = customer_data.iloc[:, 2:5].values

df_new = pd.concat([pd.DataFrame(X),Y],axis=1)
df_new.shape
list(df_new)
pd.crosstab(Y[0],Y[0])

Y

df_new.isnull().sum()

X = df_new.iloc[:,0:3]
Y  = df_new.iloc[:,3]


# split your data in to two part - train and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.20 ,random_state = 20 )

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











