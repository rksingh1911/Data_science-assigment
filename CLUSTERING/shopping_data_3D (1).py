
###############################################################################
import pandas as pd  
#import numpy as np  

customer_data = pd.read_csv('D:\\CARRER\\My_Course\\Data Science Classes\\3 Module\\2 Unsupervised\\1 Cluster Analysis\\HIERARCHEAL\\shopping_data.csv', delimiter=',') 
customer_data.shape
customer_data.head()
X = customer_data.iloc[:, 2:5].values 
X.shape
##############################################################################
%matplotlib qt
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


%matplotlib qt
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
X = 

df_new = pd.concat([pd.DataFrame(X),Y],axis=1)

pd.crosstab(Y[0],Y[0])

Y

#########################################################

clust = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(X)
    clust.append(kmeans.inertia_)
    
plt.plot(range(1, 11), clust)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertial values')
plt.show()

print(clust)

