# Profit based on the 4 independent variables
# y = b0 + b1x1 + b2x2 + b3x3 ...
# State is a word, unlike other variables, so we convert to dummy variable
# Create a new category for each column
# Always exclude 1 dummy variable to avoid the dummy variable trap

# All in: Throw all variables into the model
# Backward Elimination: 5%, all in, remove highest p value
# Forward selection: 5%, all in, select lowest p value, select with 2 variables...
# Bidirectional elimination: significance level to enter the model, and to exit the model
# All possible models: select the best one

# Backward elimination is the fastest one and is used here

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Importing the dataset
dataset = pd.read_csv('Multiple_Linear_Regression/50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Fit label encoder on the first column
LabelEncoder_X = LabelEncoder()
X[:, 3] = LabelEncoder_X.fit_transform(X[:, 3])

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],
    remainder = 'passthrough'
)
X = np.array(ct.fit_transform(X), dtype=np.float)

# Avoiding the dummy variable trap - remove 1 dummy variable
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Multiple linear regressor
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prediction of test set results
y_pred = regressor.predict(X_test)

plt.plot(y_test, color='blue')
plt.plot(y_pred, color='red')
plt.legend(['Actual profit', 'Predicted profit'])
plt.title('Predicted profit (Testing set)')
plt.ylabel('Profit')
plt.show()