# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 12:09:37 2021

@author: adity
"""

import pandas as pd
import numpy as np

train = pd.read_csv('cleaned_titanic.csv')
test = pd.read_csv('cleaned_test.csv')

train.columns
test.columns

## Drop the extra columns from both training and test set
train.drop(["Age","Embarked"],axis=1,inplace = True)
test.drop(["Age","Fare"],axis = 1,inplace = True)

## Rename the column names so that they have the same features
train.rename(columns = {"impAge":"Age","impEmbarked":"Embarked"},inplace = True)
test.rename(columns = {"impAge":"Age","impFare":"Fare"},inplace = True)

y = train["Survived"]

features = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]

X = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100,max_depth=5,random_state=1)
model.fit(X,y)
predictions = model.predict(X_test)

output = pd.DataFrame({"PassengerId":test.PassengerId,"Survived":predictions})
output.to_csv("my_submission.csv",index=False)
print("Created submission file successfully")