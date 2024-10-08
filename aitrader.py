import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
dataset = pd.read_csv('C:/Users/Hp/Downloads/GOOG.csv')
print(dataset.head())
dataset['Date'] = pd.to_datetime(dataset.Date)
print(dataset.shape)
dataset.drop('Adj Close',axis = 1, inplace = True)
print(dataset.head())
print(dataset.isnull().sum())
print(dataset.isna().any())
print(dataset.info())
print(dataset.describe())
print(len(dataset))
dataset['Open'].plot(figsize=(16,6))
plt.show()
from sklearn.model_selection import train_test_split
# Build the model:
X = dataset[['Open','High','Low','Volume']]
y = dataset['Close']
train, test = train_test_split(dataset, test_size=0.3, shuffle=False)
# Training data:
X_train = train[['Open','High','Low','Volume']]
y_train = train['Close']
# testing data
X_test = test[['Open','High','Low','Volume']]
y_test = test['Close']
print(X_train.shape)
print(X_test.shape)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
from sklearn.metrics import mean_squared_error


print(regressor.coef_)
print(regressor.intercept_)
predicted=regressor.predict(X_test)
print(X_test)
print(predicted.shape)
dframe=pd.DataFrame(y_test,predicted)
dfr=pd.DataFrame({'Actual':y_test,'Predicted':predicted})
print(dfr)
dfr.head(25)
from sklearn.metrics import confusion_matrix, accuracy_score
print(regressor.score(X_test,y_test))
import math
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,predicted))
print('Mean Squared  Error:',metrics.mean_squared_error(y_test,predicted))
print('Root Mean Squared Error:',math.sqrt(metrics.mean_squared_error(y_test,predicted)))
from sklearn.metrics import explained_variance_score
evs = explained_variance_score(y_test, y_pred)
print('Explained Variance Score:', evs)
print(regressor.score(X_test,y_test))
print('R-squared score:', regressor.score(X_test, y_test))
graph=dfr.head(20)
graph.plot(kind='bar')

plt.show()
