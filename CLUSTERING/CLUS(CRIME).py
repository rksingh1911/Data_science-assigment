"""
Created on Fri Jun 17 23:54:10 2022

@author: ROHIT SINGH
"""

#Business Problem
'''Perform Clustering(Hierarchical, Kmeans & DBSCAN) for the crime data 
and ientify the number of clusters formed and draw inferences.
'''

#Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#loading the data
df = pd.read_csv("E:\\ExcelR\\assigment\\CLUSTERING\\crime_data.csv")
df

#Exploratory Data Analysis
df.head()
df.tail()
df.shape
list(df)
type(df)
df.ndim
df.info()

#describe for all mean mode information
df.describe()

#finding null value
df.isnull()
df.isnull().sum()

#skew for normal distributaion
df.skew()  #so  some  variable skewness are normal 

#corelation
df.corr()

#Data visulazation

#Histogram
df.hist(figsize= (20,7))

#density plot 
df.plot(kind='density', subplots=True, layout=(4,5), figsize=(13,20), sharex=False, sharey=False)
plt.show()

#barplot
df.plot(kind = 'bar',subplots=True, layout=(4,5), figsize=(13,20))

#line plot
df.plot(kind = 'line',figsize = (13,20))

#pair plot
sns.pairplot(df)

#difine X 
x=df.iloc[:,1:5].values
x

#data preprocessing
from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
x_scale = SS.fit_transform(x)

#ploting the data
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(16,9)

%matplotlib qt
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(x_scale[:,0],x_scale[:,1],x_scale[:,2],x_scale[:,3])
plt.show()

#we can see there are outlier so can can perfrom dbscan
######################################################################
#DBSCAN
 
from sklearn.cluster import DBSCAN
db=DBSCAN(eps=2,min_samples=3).fit(x_scale)
db.labels_

cl=pd.DataFrame(db.labels_,columns=['Cluster'])
cl
cl['Cluster'].value_counts()

data_new=pd.concat([pd.DataFrame(x_scale),cl],axis=1)

#Noise data
nd=data_new[data_new['Cluster']==-1]
nd

#Final data without outliers
fd=data_new[data_new['Cluster']==0]
fd
data_new.mean()
fd.mean()

"from DBSCAN we got 1 cluster can be found"
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Hierarchical Clustering
#------------------------
# Dendogram
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(x_scale, method='ward'))

plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(x_scale, method='ward'))
plt.axhline(y=400, color='r', linestyle='--')

# the line cuts the dendogram at two points, so we have 2 clusters.

# Model Fitting 
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
Y = cluster.fit_predict(x_scale)

plt.figure(figsize=(16,9))
plt.scatter(x_scale[:,0],x_scale[:,1],x_scale[:,2],c=Y,cmap='rainbow')

"from Agglomerative we got 2 cluster can be found"

#==============================================================================
#NON-Hierarchical Clustering
# K means Clustering
#-------------------
from sklearn.cluster import KMeans
wcss = [] 
for i in range(1, 11): 
    km = KMeans(n_clusters = i, init = 'k-means++', random_state = 1)
    km.fit_predict(x_scale) 
    wcss.append(km.inertia_)
    print(wcss)

# Elbow plot   
plt.plot(range(1, 11), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') 
plt.show()

#we can see 4 is the drop point for the elbow so we can take 4 clusters

# Model Fitting
km = KMeans(n_clusters = 6)
Y_means = km.fit_predict(x_scale) 
Y_means

c=km.cluster_centers_
print(c)
km.inertia_

#ploting the result
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
%matplotlib qt
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(x_scale[:,0],x_scale[:,1],x_scale[:,2],x_scale[:,3])
ax.scatter(c[:,0],c[:,1],c[:,2],c[:,3],marker='*',c='Red',s=1000)


# Plotting the result
%matplotlib qt
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(x_scale[Y_means == 0,0],x_scale[Y_means == 0,1],x_scale[Y_means==0,2],x_scale[Y_means == 0,3],color='blue')
ax.scatter(x_scale[Y_means == 1,0],x_scale[Y_means == 1,1],x_scale[Y_means==1,2],x_scale[Y_means == 1,3],color='red')
ax.scatter(x_scale[Y_means == 2,0],x_scale[Y_means == 2,1],x_scale[Y_means==2,2],x_scale[Y_means == 2,3],color='orange')
ax.scatter(x_scale[Y_means == 3,0],x_scale[Y_means == 3,1],x_scale[Y_means==3,2],x_scale[Y_means == 3,3],color='green')

"""
Inference : 
Hierarchical Clustering - the line cuts the dendogram at 2 points, so 2 clusters are formed.
 
K means Clustering - The elbow shape is created at 4, i.e., our K value or an optimal number of clusters is 4.
                      with inertia 24417.023523809516
                      
DBScan - we are getting 1 cluster can be found.

#out of the all we can see Hierarchical Clustering is good 
"""





