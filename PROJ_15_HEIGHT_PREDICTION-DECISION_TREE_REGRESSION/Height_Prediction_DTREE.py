import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

dataset = pd.read_csv('dataset.csv')
# print(dataset.shape)
# print(dataset.head(10))
a = []
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
# print(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
print(x_train)
# print(x_train.shape)

for i in x_train:
    a.append(i)

print(a)
# print(x_test.shape)
model = DecisionTreeRegressor()
model.fit(x_train, y_train)
"""

x_val = np.arange(min(x_train), max(x_train), 0.01)
x_val = x_val.reshape((len(x_val), 1))
print(x_val)
plt.scatter(x_train, y_train, marker='*', color='red')
plt.plot(x_val, model.predict(x_val), color="green")
plt.figure()
plt.show()
"""

"""
y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Root mean square error :", rmse)
r2_score = r2_score(y_test, y_pred)

print("R2 score :", r2_score*100)
"""