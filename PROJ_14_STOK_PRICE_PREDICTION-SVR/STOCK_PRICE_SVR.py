import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

dataset = pd.read_csv('data.csv')
print(dataset.shape)
print(dataset.head(10))


x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]


"""
model = LinearRegression()
model.fit(x, y)
plt.scatter(x, y, marker='*', color="red")
plt.plot(x, model.predict(x))
plt.title('LinearRegression')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

modelPR = PolynomialFeatures(degree=4)
xPoly = modelPR.fit_transform(x)
modelPLR = LinearRegression()
modelPLR.fit(xPoly, y)
plt.scatter(x, y, marker='*', color='red')
plt.title('polynomial Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x, modelPLR.predict(xPoly))
plt.show()

"""

modelSVR = SVR(kernel='rbf', degree=3)
modelSVR.fit(x, y)
plt.scatter(x, y, marker='*', color="red")
plt.plot(x, modelSVR.predict(x))
plt.title('Support Vector Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

a = [[345.45]]
result = modelSVR.predict(a)
print(result)
print(modelSVR.score(x, y) * 100)