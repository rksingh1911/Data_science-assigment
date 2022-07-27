###############################################################################
import pandas as pd  
#import numpy as np  

customer_data = pd.read_csv('shopping_data.csv', delimiter=',') 
customer_data.shape
customer_data.head()
data = customer_data.iloc[:, 3:5].values 
data.shape

##############################################################################

import scipy.cluster.hierarchy as shc

# construction of Dendogram
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))  
plt.title("Customer Dendograms")  
dend = shc.dendrogram(shc.linkage(data, method='complete')) 

"""
Now we know the number of clusters for our dataset, 
the next step is to group the data points into these five clusters. 
To do so we will again use the AgglomerativeClustering
"""
## Forming a group using clusters
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')
Y = cluster.fit_predict(data)

plt.figure(figsize=(10, 7))  
plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')  

Y_clust = pd.DataFrame(Y)
Y_clust[0].value_counts()


Y = pd.DataFrame(Y)
X = customer_data.iloc[:, 2:5].values

df_new = pd.concat([pd.DataFrame(X),Y],axis=1)
df_new.shape
list(df_new)
pd.crosstab(Y[0],Y[0])


df_new.isnull().sum()

X = df_new.iloc[:,0:3]
Y  = df_new.iloc[:,3]


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



##############################################################################

##  Implementing K-Means Clustering in Python ###


"""
# import KMeans
from sklearn.cluster import KMeans

# create kmeans object
kmeans = KMeans(n_clusters=4)
# fit kmeans object to data
# create np array for data points
points = (data[:,0], data[:,1])
kmeans.fit(points)
# print location of clusters learned by kmeans object
print(kmeans.cluster_centers_)
# save new clusters for chart
y_km = kmeans.fit_predict(points)
"""

