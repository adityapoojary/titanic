# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 12:09:37 2021

@author: adity
"""

import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

from sklearn.ensemble import RandomForestClassifier

y  = train["Survived"]

features = ["Pclass","Sex","SibSp","Parch"]

X = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])

model = RandomForestClassifier(n_estimators=100,max_depth=5,random_state=1)
model.fit(X,y)
predictions = model.predict(X_test)

output = pd.DataFrame({"PassengerId":test.PassengerId,"Survived":predictions})
output.to_csv("my_submission.csv",index=False)
print("Created submission file successfully")