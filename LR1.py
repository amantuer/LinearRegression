import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the Boston Housing dataset
#boston = load_boston()

df = pd.read_csv('BostonHousing.csv')

# Assign the features and target
X = df.drop("medv", axis=1)
y = df["medv"]

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=41)

# Train the Linear Regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Predict the target on the test set
y_pred = reg.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')