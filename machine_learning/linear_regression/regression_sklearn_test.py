import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbi
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import os

FILE_PATH = os.path.dirname(__file__)
FILE_PATH = os.path.join(FILE_PATH, 'data', 'winequality.csv')
df = pd.read_csv(FILE_PATH)

X = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']].values
y = df['quality'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

df_pred = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).head(25)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

df_pred.plot(kind='bar', figsize=(10,8))
plt.show()

