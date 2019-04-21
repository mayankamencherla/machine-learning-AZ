#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 11:02:07 2019

@author: mayank
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

dataset = pd.read_csv('Data_Preprocessing/Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:, 3].values

# Handle missing data by using mean of the values
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

imputer = imputer.fit(X[:, 1:3])

X[:, 1:3] = imputer.transform(X[:, 1:3])

# Fit label encoder on the first column
LabelEncoder_X = LabelEncoder()
X[:, 0] = LabelEncoder_X.fit_transform(X[:, 0])

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Label encode the purchase column in Y
LabelEncoder_Y = LabelEncoder()
Y = LabelEncoder_Y.fit_transform(Y)
