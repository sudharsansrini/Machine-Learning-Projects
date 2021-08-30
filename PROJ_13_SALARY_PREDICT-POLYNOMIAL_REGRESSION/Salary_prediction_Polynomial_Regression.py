import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('dataset.csv')

print(dataset.shape)
# print(dataset.head(5))

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

print(x.shape)
print(y.shape)

"""
modelLR = LinearRegression()
modelLR.fit(x, y)

plt.scatter(x, y, marker='*', color='red')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.plot(x, modelLR.predict(x))
plt.show()
"""

modelPR = PolynomialFeatures(degree=4)
xPoly = modelPR.fit_transform(x)
modelPLR = LinearRegression()
modelPLR.fit(xPoly, y)

plt.scatter(x, y, marker='*', color='red')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.plot(x, modelPLR.predict(xPoly))
plt.show()

x = [[2.5]]
Salary_Pred = modelPLR.predict(modelPR.fit_transform(x))
print("The Salary of level {0} person's salary is {1}".format(x, Salary_Pred))
# print("Accuracy Score is {0}".format(modelPLR.score(x, Salary_Pred)))
