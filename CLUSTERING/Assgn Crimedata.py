# -*- coding: utf-8 -*-
"""
Created on Mon May  2 16:14:29 2022

@author: Suklesh
"""
#Performing Clustering on Crime dataset
#a)KMeans Clustering
import pandas as pd
import numpy as np
data=pd.read_csv('F:\\ExcelR\\Assignments\\Clustering\\crime_data.csv')
data.shape
list(data)
data.dtypes
X=data.iloc[:,1:5].values
X
list(X)

#Standardizing the data
from sklearn.preprocessing import StandardScaler
X_scale=StandardScaler().fit_transform(X)
X_scale

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(16,9)

%matplotlib qt
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(X_scale[:,0],X_scale[:,1],X_scale[:,2],X_scale[:,3])
plt.show()

#Initializing KMeans
from sklearn.cluster import KMeans
km=KMeans(n_clusters=5).fit(X_scale)
lab=km.predict(X_scale)
type(lab)

c=km.cluster_centers_
km.inertia_

%matplotlib qt
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(X_scale[:,0],X_scale[:,1],X_scale[:,2],X_scale[:,3])
ax.scatter(c[:,0],c[:,1],c[:,2],c[:,3],marker='*',c='Red',s=1000)

clust=[]
for i in range(1,9,1):
    km=KMeans(n_clusters=i).fit(X_scale)
    km.inertia_
    clust.append(km.inertia_)

#Elbow plot
plt.plot(range(1,9),clust)
plt.title('Elbow Plot')
plt.xlabel('No of Clusters')
plt.ylabel('Cluster Inertia values')
plt.show()
###################################################################################################
#b)Hierarchial Clustering
#Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering
ac=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='complete')
ac.fit_predict(X_scale)

plt.figure(figsize=(16,9))
plt.scatter(X_scale[:,0],X_scale[:,1],X_scale[:,2],c=ac.labels_,cmap='rainbow')

#Dendogram
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize=(16,9))
plt.title('Dendogram')
dend=shc.dendrogram(shc.linkage(X,method='complete'))
###################################################################################################
#c)DBScan 
from sklearn.cluster import DBSCAN
db=DBSCAN(eps=2,min_samples=3).fit(X_scale)
db.labels_

cl=pd.DataFrame(db.labels_,columns=['Cluster'])
cl
cl['Cluster'].value_counts()

data_new=pd.concat([pd.DataFrame(X_scale),cl],axis=1)

#Noise data
nd=data_new[data_new['Cluster']==-1]
nd

#Final data without outliers
fd=data_new[data_new['Cluster']==0]
fd
data_new.mean()
fd.mean()










