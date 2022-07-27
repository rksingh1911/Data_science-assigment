
import pandas as pd

##########################

df = pd.read_csv("100_data_names.csv")
df

list(df)

X1 = pd.DataFrame(df)
X1.shape
X1.head()

##########################
# load decomposition to do PCA analysis with sklearn
from sklearn.decomposition import PCA
PCA()
pca = PCA(svd_solver='full')

pc = pca.fit_transform(X1)
pca.explained_variance_ratio_
sum(pca.explained_variance_ratio_)

pc.shape
pd.DataFrame(pc).head()
type(pc)

pc_df = pd.DataFrame(data = pc , columns = ['P0C1', 'P0C2','P0C3','P0C4','P0C5','P0C6','P0C7','P0C8','P096','P0C10','P1C1', 'P1C2','P1C3','P1C4','P1C5','P1C6','P1C7','P1C8','P196','P1C10','P2C1', 'P2C2','P2C3','P2C4','P2C5','P2C6','P2C7','P2C8','P296','P2C10','P3C1', 'P3C2','P3C3','P3C4','P3C5','P3C6','P3C7','P3C8','P396','P3C10','P4C1', 'P4C2','P4C3','P4C4','P4C5','P4C6','P4C7','P4C8','P496','P4C10','P5C1', 'P5C2','P5C3','P5C4','P5C5','P5C6','P5C7','P5C8','P596','P5C10','P6C1', 'P6C2','P6C3','P6C4','P6C5','P6C6','P6C7','P6C8','P696','P6C10','P7C1', 'P7C2','P7C3','P7C4','P7C5','P7C6','P7C7','P7C8','P796','P7C10','P8C1', 'P8C2','P8C3','P8C4','P8C5','P8C6','P8C7','P8C8','P896','P8C10','P9C1', 'P9C2','P9C3','P9C4','P9C5','P9C6','P9C7','P9C8','P996','P9C10'])
pc_df.head()
pc_df.shape
type(pc_df)

pc_df.to_csv("PC100.csv")


"""
variance explained by each principal component is called Scree plot.
"""
import seaborn as sns
df = pd.DataFrame({'var':pca.explained_variance_ratio_,
                  'PC':['P1C1', 'P1C2','P1C3','P1C4','P1C5','P1C6','P1C7','P1C8','P196','P1C10','P2C1', 'P2C2','P2C3','P2C4','P2C5','P2C6','P2C7','P2C8','P296','P2C10','P3C1', 'P3C2','P3C3','P3C4','P3C5','P3C6','P3C7','P3C8','P396','P3C10','P4C1', 'P4C2','P4C3','P4C4','P4C5','P4C6','P4C7','P4C8','P496','P4C10','P5C1', 'P5C2','P5C3','P5C4','P5C5','P5C6','P5C7','P5C8','P596','P5C10','P6C1', 'P6C2','P6C3','P6C4','P6C5','P6C6','P6C7','P6C8','P696','P6C10','P7C1', 'P7C2','P7C3','P7C4','P7C5','P7C6','P7C7','P7C8','P796','P7C10','P8C1', 'P8C2','P8C3','P8C4','P8C5','P8C6','P8C7','P8C8','P896','P8C10','P9C1', 'P9C2','P9C3','P9C4','P9C5','P9C6','P9C7','P9C8','P996','P9C10','P0C1', 'P0C2','P0C3','P0C4','P0C5','P0C6','P0C7','P0C8','P096','P0C10']})
sns.barplot(x='PC',y="var", data=df, color="c");
            
#########################

pca = PCA(n_components=5)
pc = pca.fit_transform(X1)
pc.shape

pc_df = pd.DataFrame(data = pc , columns = ['PC1', 'PC2','PC3','PC4','PC5'])
pc_df.head()

pca.explained_variance_ratio_


"""variance explained by each principal component is called Scree plot.
"""
import seaborn as sns
df = pd.DataFrame({'var':pca.explained_variance_ratio_,
                  'PC':['PC1','PC2','PC3','PC4','PC5']})
sns.barplot(x='PC',y="var", data=df, color="c");

#########################333333333333333333333333
pc_df
list(pc_df)
pc_df.isnull().sum()
''''
from sklearn.preprocessing import StandardScaler
stscaler = StandardScaler().fit(pc_df)
X = stscaler.transform(pc_df)

from sklearn.cluster import DBSCAN
DBSCAN()
dbscan = DBSCAN(eps=3, min_samples=7)
dbscan.fit(pc_df)

#Noisy samples are given the label -1.
dbscan.labels_

cl=pd.DataFrame(dbscan.labels_,columns=['cluster'])
cl
cl['cluster'].value_counts()

clustered = pd.concat([df,cl],axis=1)

noisedata = clustered[clustered['cluster']==-1]
finaldata = clustered[clustered['cluster']==0]

clustered'''

import scipy.cluster.hierarchy as shc

# construction of Dendogram
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))  
plt.title("Customer Dendograms")  
dend = shc.dendrogram(shc.linkage(pc_df, method='complete')) 

"""
Now we know the number of clusters for our dataset, 
the next step is to group the data points into these five clusters. 
To do so we will again use the AgglomerativeClustering
"""
## Forming a group using clusters
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')
Y = cluster.fit_predict(pc_df)

#need to check
#plt.figure(figsize=(10, 7))  
#plt.scatter(pc_df[:,], pc_df[:,1], pc_df[:,2], c=cluster.labels_, cmap='rainbow')  

Y_clust = pd.DataFrame(Y)
Y_clust[0].value_counts()


Y = pd.DataFrame(Y)
#X = pc_df.iloc[:, 2:5].values

df_new = pd.concat([pd.DataFrame(pc_df),Y],axis=1)
df_new.shape
list(df_new)
pd.crosstab(Y[0],Y[0])

df_new.isnull().sum()

X = df_new.iloc[:,0:5]
Y  = df_new.iloc[:,5]


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




