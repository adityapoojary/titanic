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

