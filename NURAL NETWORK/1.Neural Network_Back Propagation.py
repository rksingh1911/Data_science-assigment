# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 20:15:18 2022

@author: Hi
"""

!pip install tensorflow
!pip install keras

# Create your first MLP in Keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy

# fix random seed for reproducibility
# seed = 7
# numpy.random.seed(seed)
# load pima indians dataset
# dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
import pandas as pd
df = pd.read_csv("pima-indians-diabetes.csv",delimiter=",")
df.head()

# split into input (X) and output (Y) variables
X = df.iloc[:,0:8]
X.shape
Y = df.iloc[:,8]


# create model
model = Sequential()
model.add(Dense(12, input_dim=8,  activation='relu')) #input layer
model.add(Dense(1, activation='sigmoid')) #output layer


#model.add(Dense(8,  activation='relu')) #2nd layer
#![image.png](attachment:image.png)

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#768*0.67

# Fit the model
history = model.fit(X, Y, validation_split=0.33, epochs=250, batch_size=10)

# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Visualize training history

# list all data in history
history.history.keys()


# summarize history for accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


