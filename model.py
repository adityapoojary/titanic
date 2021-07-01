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
test_X = pd.get_dummies(test[features])

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier

## For Tuning hyperparameters
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

## Splitting the data to create test and training sets
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

## Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)

accuracy_score(y_test, y_pred)
print(classification_report(y_test,y_pred))

## KNN
knn = KNeighborsClassifier(metric='manhattan',n_neighbors=5,weights='uniform')
n_neighbors = range(1, 21, 2)
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']
# define grid search
grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=knn, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

accuracy_score(y_test, y_pred)
print(classification_report(y_test,y_pred))

## Logistic Regression
lr = LogisticRegression(C=10,penalty='l2',solver='newton-cg')
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=lr, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

accuracy_score(y_test, y_pred)
print(classification_report(y_test,y_pred))

## Random Forest Classifier
rf = RandomForestClassifier(max_features='log2',n_estimators=10)
n_estimators = [10, 100, 1000]
max_features = ['sqrt', 'log2']
# define grid search
grid = dict(n_estimators=n_estimators,max_features=max_features)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=rf, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)

accuracy_score(y_test, y_pred)
print(classification_report(y_test,y_pred))

## XGboost
xgb = XGBClassifier()
xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)

accuracy_score(y_test, y_pred)
print(classification_report(y_test,y_pred))

## Stochastic Gradient Boosting
gbc = GradientBoostingClassifier(learning_rate=0.01,max_depth=3,n_estimators=1000,subsample=0.7)
n_estimators = [10, 100, 1000]
learning_rate = [0.001, 0.01, 0.1]
subsample = [0.5, 0.7, 1.0]
max_depth = [3, 7, 9]
# define grid search
grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=gbc, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

gbc.fit(X_train,y_train)
y_pred = gbc.predict(X_test)

accuracy_score(y_test, y_pred)
print(classification_report(y_test,y_pred))

predictions = gbc.predict(test_X)

## Creating output file for submission
output = pd.DataFrame({"PassengerId":test.PassengerId,"Survived":predictions})
output.to_csv("my_submission.csv",index=False)
print("Created submission file successfully")