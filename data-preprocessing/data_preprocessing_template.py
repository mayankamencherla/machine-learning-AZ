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
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

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

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],
    remainder = 'passthrough'
)
X = np.array(ct.fit_transform(X), dtype=np.float)

# Label encode the purchase column in Y
LabelEncoder_Y = LabelEncoder()
Y = LabelEncoder_Y.fit_transform(Y)

# Split into testing and training sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

