"""
Created on Fri Jun 17 20:33:07 2022

@author: ROHIT SINGH
"""
#Business Problem
'''Perform clustering (hierarchical,K means clustering and DBSCAN) 
for the airlines data to obtain optimum number of clusters. 
Draw the inferences from the clusters obtained.
'''
#Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#loading the data
df = pd.read_excel("E:\\ExcelR\\assigment\\CLUSTERING\\EastWestAirlines.xlsx",sheet_name='data')
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
df.skew()  #so  some  variable skewness are normal and high as well

#corelation
df.corr()

#data cleaning
df.drop(['ID#'],axis=1,inplace=True)

x=df.iloc[:,1:12].values
x.shape
list(x)

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

#data preprocessing
from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
x_scale = SS.fit_transform(x)

#================================================================
"we are doing very frist DBSCAN because from there we can remove noise (ouliar) as well"
# DBSCAN
from sklearn.cluster import DBSCAN
db=DBSCAN(eps=1,min_samples=7)
labels = db.fit_predict(x_scale)
cl=pd.DataFrame(db.labels_,columns=['Cluster'])
cl
cl['Cluster'].value_counts()
df_new = pd.concat([pd.DataFrame(x_scale),cl],axis=1)

#Noise data
nd1 = df_new[df_new['Cluster']==2]
nd2 = df_new[df_new['Cluster']==3]
nd1.shape
nd2.shape

#Final data without outliers
fd1 =df_new[df_new['Cluster']==0]
fd2 =df_new[df_new['Cluster']==1]
fd3 =df_new[df_new['Cluster']==-1]
fd = pd.concat([fd1,fd2,fd3],axis=0)
fd.shape
df_new.mean()
fd.mean()
type(fd)

"from DBSCAN we got 3 cluster can be found"
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Hierarchical Clustering
# Dendogram
import scipy.cluster.hierarchy as shc
plt.figure(figsize = (20,7))
plt.title("Dendorograms")
dendrogram = shc.dendrogram(shc.linkage(x_scale,method = 'ward'))

# Number of clusters
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
z = shc.linkage(x_scale, method='ward')
dendrogram = shc.dendrogram(z)
plt.axhline(y=80,color='red', linestyle='--')
plt.show()

# the line cuts the dendogram by 5 points, so we have 5 clusters

# Model Fitting 
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='complete')
Y = cluster.fit_predict(x_scale)

# Plotting resulting clusters

# Function for creating datapoints in the form of a circle
import matplotlib
import math
def PointsInCircum(r,n=100):
    return [(math.cos(2*math.pi/n*x)*r+np.random.normal(-30,30),math.sin(2*math.pi/n*x)*r+np.random.normal(-30,30)) for x in range(1,n+1)]

# Creating data points in the form of a circle
x_scale=pd.DataFrame(PointsInCircum(2000,3000))
x_scale=x_scale.append(PointsInCircum(800,1800))
x_scale=x_scale.append(PointsInCircum(100,800))
# Adding noise to the dataset
x_scale=x_scale.append([(np.random.randint(-600,600),np.random.randint(-600,600)) for i in range(300)])

plt.figure(figsize=(10,10))
plt.scatter(x_scale[0],x_scale[1],c = Y,s=15)
plt.title('Hierarchical Clustering',fontsize=20)
plt.xlabel('Feature 1',fontsize=14)
plt.ylabel('Feature 2',fontsize=14)
plt.show()
#we can observ that we can make 5 cluster 

#--------------------------------------------------------------------
# K means clustering
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
#we can see 6 is the drop point for the elbow so we can take 6 clusters

# Model Fitting
km = KMeans(n_clusters = 6)
Y_means = km.fit_predict(x_scale) 
Y_means

c=km.cluster_centers_
print(c)
km.inertia_


#ploting the result
#matplotlib qt
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
%matplotlib qt
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(x_scale[:,0],x_scale[:,1],x_scale[:,2])
ax.scatter(c[:,0],c[:,1],c[:,2],marker='*',c='Red',s=1000)

#we can see here we got 5 cluset here 

"""
Inference : 
Hierarchical Clustering - the line cuts the dendogram at 5 points, so 5 clusters are formed.
 
K means Clustering - The elbow shape is created at 6, i.e., our K value or an optimal number of clusters is 6.
                      with inertia 24417.023523809516
                      
DBScan - we are getting 3 cluster can be found.

#out of the all we can K means Clustering is good 
"""



















