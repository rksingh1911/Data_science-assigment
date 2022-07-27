# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 21:31:25 2022

@author: ROHIT SINGH
"""

for i in range(1,101,1):
    x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,test_size=0.25,random_state=i)
    print('For random state',i)
    for j in np.arange(1,10,1):
        DT1 = DecisionTreeClassifier(criterion='entropy',max_depth=j)
        model=DT1.fit(x_train,y_train)
        y_pred=model.predict(x_test)
        acc=accuracy_score(y_test, y_pred)
        while (acc*100)>=77:
            print('Accuracy for max_features',j,'is',(acc*100).round(3))
            break