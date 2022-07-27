"""
Created on Mon Apr 25 12:43:06 2022
"""

##########################################################
import numpy as np
import pandas as pd
df = pd.read_csv("Boston.csv")
df.shape

X = df.iloc[:,1:14]
list(X)

Y = df.iloc[:,14]
Y
###########

from sklearn.model_selection import train_test_split
X_train,X_test, Y_train, Y_test  = train_test_split(X,Y, test_size = 0.20, random_state = 100)

from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(X_train,Y_train)

y_pred = dtr.predict(X_test)

print(f"Decision tree has {dtr.tree_.node_count} nodes with maximum depth covered up to {dtr.tree_.max_depth}")

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y_test,y_pred)

RMSE = np.sqrt(mse)
RMSE

#------------------------------------------------------------------
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(learning_rate=0.1,n_estimators=100) # lr = 0.1, est = 100

gbr.fit(X_train,Y_train)

y_pred = gbr.predict(X_test)

mse = mean_squared_error(Y_test,y_pred)
RMSE = np.sqrt(mse)
RMSE
#------------------------------------------------------------------

from sklearn.ensemble import AdaBoostRegressor
AB = AdaBoostRegressor(base_estimator=dtr,n_estimators=100) 

AB.fit(X_train,Y_train)
y_pred = AB.predict(X_test)

mse = mean_squared_error(Y_test,y_pred)
RMSE = np.sqrt(mse)
RMSE
#------------------------------------------------------------------


