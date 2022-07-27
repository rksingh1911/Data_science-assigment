"""
Created on Wed Jun 22 18:22:16 2022

@author: ROHIT SINGH
"""

#Business Problem
'''Perform Principal component analysis and perform clustering using 
first 3 principal component scores (both heirarchial and k mean clustering
(scree plot or elbow curve) and obtain optimum number of clusters and check
whether we have obtained same number of clusters with the original data.
'''

#Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#loading the data
df = pd.read_csv("E:\\ExcelR\\assigment\\PCA\\wine.csv")
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

#defining x  and y variable
X=df.iloc[:,1:15]
X.head()
list(X)

Y=df['Type']
Y.head
Y.value_counts()
type(Y)

#Preproceesing the data
from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
X_scale=SS.fit_transform(X)

#Decomposition
from sklearn.decomposition import PCA
pca=PCA(svd_solver='full')
pc=pca.fit_transform(X_scale)
pca.explained_variance_ratio_
sum(pca.explained_variance_ratio_)
pc.shape
type(pc)

pc_new=pd.DataFrame(data=pc,columns=['PC_1','PC_2','PC_3','PC_4','PC_5','PC_6','PC_7','PC_8','PC_9','PC_10','PC_11','PC_12','PC_13'])
pc_new.head()
pc_new.to_csv("E:\\ExcelR\\assigment\\PCA\\pca.csv",header=True)

import seaborn as sns
data_new=pd.DataFrame({'var':pca.explained_variance_ratio_,'PC':['pc_1','pc_2','pc_3','pc_4','pc_5','pc_6','pc_7','pc_8','pc_9','pc_10','pc_11','pc_12','pc_13']})
sns.barplot(x='PC',y='var',data=data_new,color='blue')
######################################################################################################################
#Since almost 66% of the data is covered in first 3 bars as shown in the bar plot, we try by taking first 3 columns as X (as mentioned in the problem)
#Decomposition

from sklearn.decomposition import PCA
pca1=PCA(n_components=3)
pc1=pca1.fit_transform(X_scale)
pca1.explained_variance_ratio_
sum(pca1.explained_variance_ratio_)
pc1.shape
type(pc1)

pc1_new=pd.DataFrame(data=pc1,columns=['PC_1','PC_2','PC_3'])
type(pc1_new)
pc1_new.head()
pc1_new.to_csv("E:\\ExcelR\\assigment\\PCA\\\\pca1.csv",header=True)


import seaborn as sns
data1_new=pd.DataFrame({'var':pca1.explained_variance_ratio_,'PC':['pc_1','pc_2','pc_3']})
sns.barplot(x='PC',y='var',data=data1_new,color='red')

#-----------------------------------------------------------------------------
# KMeans without PCA

X1=pc1_new.iloc[:,0:3]
clust=[]
for i in range(1,10,1):
    km=KMeans(n_clusters=i).fit(X1)
    km.inertia_
    clust.append(km.inertia_)
print(clust)

# Elbow plot
plt.plot(range(1,10),clust)
plt.title("Elbow plot")
plt.xlabel('No of Cluster')
plt.ylabel('Cluster inertia values')
plt.show()
    
# The elbow shape is created at 3, i.e., our K value or an optimal number of clusters is 3.

# Model Fitting
km = KMeans(n_clusters = 3)
Y_pred_N = km.fit_predict(X1) 
Y_pred_N
Y_pred_N=pd.DataFrame(Y_pred_N)
Y_pred_N.value_counts()
type(Y_pred_N)


c1=km.cluster_centers_
c1.shape
km.inertia_

# plotting the results
%matplotlib qt
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(16,9)

from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(X1.iloc[:, 0],X1.iloc[:, 1],X1.iloc[:, 2])
ax.scatter(c[:, 0], c[:, 1], c[:, 2],marker='*', c='Red', s=1000) # S is star size, c= * color
plt.show()

'''
Original dataset contains the type as 1,2,3 but here python has given the type as 0,1,2 under Y_pred
To compare between the actual Y variable and predicted Y variable we need to convert those values
'''
Y_pred_new_=[]
for i in range(0,178,1):
    if Y_pred_N.iloc[i,0]==0:
        Y_pred_new_.append(1)
    elif Y_pred_N.iloc[i,0]==1:
        Y_pred_new_.append(2)
    else:
        Y_pred_new_.append(3)
Y_pred_df_=pd.DataFrame(Y_pred_new_)
Y_pred_df_.value_counts()

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(Y, Y_pred_df_)
cm
acc_=accuracy_score(Y, Y_pred_df_)
acc_

#===================================================================================
# Hierarchical with PCA
#Construction of Dendogram
X2=pc1_new.iloc[:,0:3].values
type(X2)
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize=(16,9))
plt.title('Dendrogram')
dend=shc.dendrogram(shc.linkage(X2,method='complete'))

 
from sklearn.cluster import AgglomerativeClustering
ac=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='complete')
Y_pred1=ac.fit_predict(X2)
Y_pred1=pd.DataFrame(Y_pred1)
Y_pred1.shape
Y_pred1.value_counts()

plt.figure(figsize=(16,9))
plt.scatter(X2[:,0],X2[:,1],X2[:,2],c=Y_pred1,cmap='rainbow')

%matplotlib qt
plt.rcParams['figure.figsize']=(16,9)

from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(X2[:, 0],X2[:, 1],X2[:, 2])
plt.show()

Y=df['Type']
Y.value_counts()
type(Y)

'''
Original dataset contains the type as 1,2,3 but here python has given the type as 0,1,2 under Y_pred
To compare between the actual Y variable and predicted Y (Y_pred1) variable we need to convert those values
'''

Y_pred_new1=[]
for i in range(0,178,1):
    if Y_pred1.iloc[i,0]==0:
        Y_pred_new1.append(1)
    elif Y_pred1.iloc[i,0]==1:
        Y_pred_new1.append(2)
    else:
        Y_pred_new1.append(3)
        
Y_pred_df1=pd.DataFrame(Y_pred_new1)
Y_pred_df1.value_counts()

from sklearn.metrics import confusion_matrix,accuracy_score
cm1=confusion_matrix(Y, Y_pred_df1)
cm1
acc1=accuracy_score(Y, Y_pred_df1)
acc1
########################################################################
# Hierarchical without PCA
# Dendogram
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(X, method='ward'))

# Number of clusters
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
z = shc.linkage(X, method='ward')
dend = shc.dendrogram(z)
plt.axhline(y=2137,color='r', linestyle='--')
plt.show()

# the line cuts the dendogram at 3 points, so we have 3 clusters

# Model Fitting 
from sklearn.cluster import AgglomerativeClustering
ac=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='complete')
Y_pred2=ac.fit_predict(X_scale)
Y_pred2=pd.DataFrame(Y_pred2)
Y_pred2.shape
Y_pred2.value_counts()

plt.figure(figsize=(16,9))
plt.scatter(X_scale[:,0],X_scale[:,1],X_scale[:,2],c=Y_pred2,cmap='rainbow')

%matplotlib qt
plt.rcParams['figure.figsize']=(16,9)

from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(X_scale[:, 0],X_scale[:, 1],X_scale[:, 2])
plt.show()

Y=df['Type']
Y.value_counts()
type(Y)

'''
Original dataset contains the type as 1,2,3 but here python has given the type as 0,1,2 under Y_pred
To compare between the actual Y variable and predicted Y (Y_pred1) variable we need to convert those values
'''
Y_pred_new2=[]
for i in range(0,178,1):
    if Y_pred2.iloc[i,0]==0:
        Y_pred_new2.append(1)
    elif Y_pred2.iloc[i,0]==1:
        Y_pred_new2.append(2)
    else:
        Y_pred_new2.append(3)
        
Y_pred_df2=pd.DataFrame(Y_pred_new2)
Y_pred_df2.value_counts()

from sklearn.metrics import confusion_matrix,accuracy_score
cm2=confusion_matrix(Y, Y_pred_df2)
cm2
acc2=accuracy_score(Y, Y_pred_df2)
acc2

"""
Inference:

    K Means:
    In K means both pca and with pca gives equal number of clusters i.e., 3
    but accuracy scores are differnt 
    K Means with pca accuracy = 0.34831460674157305
    K Means accuracy          = 0.016853932584269662
    So, here K Means with PCA performs well.
    
    Hierarchical:  
    Hierarchical with PCA gives 5 clusters and the visualisation result is also good.
    Hierarchical without PCA gives 3 clusters and the data points are scsttered.
    Accuracy with PCA = 0.11797752808988764
    Accuracy without PCA = 0.8370786516853933
    So, Hierarchical without PCA performs good.
    
    