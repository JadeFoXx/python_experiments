from linear_model import LinearModel

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

FILE_PATH = os.path.dirname(__file__)
FILE_PATH = os.path.join(FILE_PATH, 'data', 'gpa_sat.csv')
data = pd.read_csv(FILE_PATH)
scalar = 1000
data = data / scalar

train_data = data.sample(n=50)
train_data_x = train_data['SAT'].to_numpy()
train_data_y = train_data['GPA'].to_numpy()

test_data = data.drop(train_data.index)
test_data_x = test_data['SAT'].to_numpy()
test_data_y = test_data['GPA'].to_numpy()

data = data * scalar

linear_model = LinearModel()
linear_model.fit(train_data_x, train_data_y, 'least_squares')
pred_data_y = linear_model.predict(test_data_x)

fig1, ax1 = plt.subplots()
fig1.canvas.set_window_title('least_squares')
result_data = pd.DataFrame({'SAT': test_data_x, 'GPA': pred_data_y}, columns=['SAT', 'GPA'])
result_data = result_data * scalar
result_data.plot(kind='line', x='SAT', y='GPA', color='blue', ax=ax1)
data.plot(kind='scatter', x='SAT', y='GPA', color='red', ax=ax1)

print(linear_model.coefficients())

linear_model.reset()
linear_model.fit(train_data_x, train_data_y, 'gradient_descent')
pred_data_y = linear_model.predict(test_data_x)

fig2, ax2 = plt.subplots()
fig2.canvas.set_window_title('gradient_descent')
result_data = pd.DataFrame({'SAT': test_data_x, 'GPA': pred_data_y}, columns=['SAT', 'GPA'])
result_data = result_data * scalar
result_data.plot(kind='line', x='SAT', y='GPA', color='blue', ax=ax2)
data.plot(kind='scatter', x='SAT', y='GPA', color='red', ax=ax2)

print(linear_model.coefficients())

plt.show()