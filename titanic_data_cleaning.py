# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 11:04:46 2021

@author: adity
"""
import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.head()
test.head()

train.columns

# Proportion of passengers
len(train[train.Sex=='female'])/len(train)
len(train[train.Sex=='male'])/len(train)

# Percentage of female survived
rate_women=sum(train[train.Sex=='female']['Survived'])/len(train[train.Sex=='female'])
print("Percentage of Women survived = {}".format(round(rate_women*100,2)))

# Percentage of male survived
rate_men=sum(train[train.Sex=='male']['Survived'])/len(train[train.Sex=='male'])
print("Percentage of Men survived = {}".format(round(rate_men*100,2)))

train.info()

train.describe()

train.Age.value_counts

# For Training Data
## Filling Missing Age
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan,strategy='mean')

train["Age"].isna().sum()

imp_mean = imp_mean.fit(train["Age"].values.reshape(-1,1))

train["impAge"] = imp_mean.transform(train["Age"].values.reshape(-1,1))

## Filling missing Embarked with mode
 
train["impEmbarked"] = train["Embarked"].fillna(train["Embarked"].mode()[0])

train.to_csv('cleaned_train.csv',index=False)

# For Test Data
test.info()
from sklearn.impute import SimpleImputer
## For Age
imp_mean = SimpleImputer(missing_values=np.nan,strategy='mean')
test["Age"].isna().sum()
imp_mean = imp_mean.fit(test["Age"].values.reshape(-1,1))
test["impAge"] = imp_mean.transform(test["Age"].values.reshape(-1,1))

## For fare
imp_fare = SimpleImputer(missing_values=np.nan,strategy='mean')
imp_fare = imp_fare.fit(test["Fare"].values.reshape(-1,1))
test["impFare"] = imp_fare.transform(test["Fare"].values.reshape(-1,1))

test.to_csv('cleaned_test.csv',index = False)
