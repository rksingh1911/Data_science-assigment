"""
Created on Mon Sep 27 11:32:17 2021

Apriori
Importing the libraries
"""

#pip install apyori

#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
dataset.shape
str(dataset.values[0,0])
str(dataset.values[0,19])

transactions = []
for i in range(0, 7501):
  transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
transactions
len(transactions)

type(transactions)

#Training the Apriori model on the dataset
# pip install apyori
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

# Visualising the results
# Displaying the first results coming directly from the output of the apriori function
results = list(rules)
results
len(results)
type(results)

results[0]
results[0][0]
results[0][1]
results[0][2]
results[0][2][0][0]
results[0][2][0][1]
results[0][2][0][2]
results[0][2][0][3]





#Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])


#Displaying the results non sorted
resultsinDataFrame

#Displaying the results sorted by descending lifts
resultsinDataFrame.nlargest(n = 10, columns = 'Lift')



