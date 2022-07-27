
import pandas as pd  
import numpy as np  
# import matplotlib.pyplot as plt  
# %matplotlib inline

df = pd.read_csv("D:\\CARRER\\My_Course\\Data Science Classes\\3 Module\\1 Supervised\\10 Decision Tree\\2 GINI\\sales.csv")  
df.shape
df.head() 
df.describe()
type(df)


###############################################################
# Label encode
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df['ShelveLoc'] = label_encoder.fit_transform(df['ShelveLoc'])
df['Urban'] = label_encoder.fit_transform(df['Urban'])
df['US'] = label_encoder.fit_transform(df['US'])
df.head()

#X = dataset.drop('high', axis=1)  #Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’)
#X = dataset.iloc[:,1:5]  #Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’)
X = df.iloc[:,1:11]  #Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’)
Y = df['high']


X.head()
Y.head()
list(X)
###############################################################

from sklearn.model_selection import train_test_split  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20,random_state=42,stratify=Y) 

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
X_train.ndim
Y_train.ndim

from sklearn.tree import DecisionTreeClassifier
#classifier = DecisionTreeClassifier() # By default gini will be added
#classifier = DecisionTreeClassifier(criterion='entropy',max_depth=9) # By default gini will be added
classifier = DecisionTreeClassifier(max_depth=3) # best fit
# classifier = DecisionTreeClassifier(criterion='entropy',max_depth = 8)


classifier.fit(X_train, Y_train)
print(f'Decision tree has {classifier.tree_.node_count} nodes with maximum depth {classifier.tree_.max_depth}.')

classifier.tree_.node_count # counting the number of nodes
classifier.tree_.max_depth # number of levels

Y_pred = classifier.predict(X_test) 

df=pd.DataFrame({'Actual':Y_test, 'Predicted':Y_pred})  
df

from sklearn import metrics
cm = metrics.confusion_matrix(Y_test, Y_pred)
print(cm)  
metrics.accuracy_score(Y_test,Y_pred).round(2)

# from sklearn.metrics import classification_report
#print(classification_report(Y_test, Y_pred)) 

###############################################################

# conda install -c anaconda graphviz
# pip install graphviz


from sklearn import tree
import graphviz 
dot_data = tree.export_graphviz(classifier, out_file=None, 
                    filled=True, rounded=True,  
                    special_characters=True)  
graph = graphviz.Source(dot_data)  
graph

###############################################################

# Create two lists for training and test accuracies
training_accuracy = []
test_accuracy = []

# Define a range of 1 to 10 (included) neighbors to be tested
settings = range(1,14)

# Loop with the KNN through the different number of neighbors to determine the most appropriate (best)
for max_depth in settings:
    clf = DecisionTreeClassifier(max_depth=max_depth,criterion='entropy')
    clf.fit(X_train, Y_train)
    training_accuracy.append(clf.score(X_train, Y_train))
    test_accuracy.append(clf.score(X_test, Y_test))
    
print(training_accuracy)
print(test_accuracy)


# Visualize results - to help with deciding which n_neigbors yields the best results (n_neighbors=6, in this case)
import matplotlib.pyplot as plt
%matplotlib qt
plt.plot(settings, training_accuracy, label='Accuracy of the training set')
plt.plot(settings, test_accuracy, label='Accuracy of the test set')
plt.ylabel('Accuracy')
plt.xlabel('Max_depth')
plt.legend()


classifier.fit(X_train, Y_train)
print(f'Decision tree has {classifier.tree_.node_count} nodes with maximum depth {classifier.tree_.max_depth}.')




