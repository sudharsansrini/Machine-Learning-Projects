import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

dataset = pd.read_csv('data.csv')

print(dataset.shape)
# print(dataset.head(10))

# print(dataset.columns[dataset.isna().any()])

dataset.hours = dataset.hours.fillna(dataset.hours.mean())

print(dataset.head(10))

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

model = LinearRegression()
model.fit(x, y)

figure = plt.figure()
plot = figure.add_subplot(projection='3d')
plot.scatter(dataset['hours'], dataset['age'], dataset['internet'], marker='*', color='red')
plt.show()


a= [[6.98, 17, 1]]
result = model.predict(a)
print(result)
print("Accuracy ", model.score(x, y) * 100)