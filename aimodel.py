import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv('C:/Users/Hp/Downloads/GOOG.csv')

# Convert Date column to datetime
dataset['Date'] = pd.to_datetime(dataset.Date)

# Drop Adj Close column
dataset.drop('Adj Close',axis=1,inplace=True)

# Check for missing values
print(dataset.isnull().sum())

# Print dataset info
print(dataset.info())

# Plot Open column
dataset['Open'].plot(figsize=(16,6))
plt.show()

# Split the dataset into training and testing sets
train, test = dataset[:1500], dataset[1500:]

# Training data
X_train = train[['Open', 'High', 'Low', 'Volume']]
y_train = train['Close']

# Testing data
X_test = test[['Open', 'High', 'Low', 'Volume']]
y_test = test['Close']

# Fit the Linear Regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the values for testing data
y_pred = regressor.predict(X_test)

# Print the model's coefficients and intercept
print(regressor.coef_)
print(regressor.intercept_)

# Print the mean squared error
mse = metrics.mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# Print the mean absolute error
mae = metrics.mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)

# Print the root mean squared error
rmse = np.sqrt(mse)
print('Root Mean Squared Error:', rmse)

# Print the explained variance score
evs = metrics.explained_variance_score(y_test, y_pred)
print('Explained Variance Score:', evs)

# Print the R-squared score
r2 = metrics.r2_score(y_test, y_pred)
print('R-squared score:', r2)

# Plot the actual and predicted values
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df.head(20).plot(kind='bar')
plt.show()
