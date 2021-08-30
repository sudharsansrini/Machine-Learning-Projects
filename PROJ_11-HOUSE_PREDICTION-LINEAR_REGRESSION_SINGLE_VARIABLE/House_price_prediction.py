import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

dataset = pd.read_csv('dataset.csv')
print(dataset.shape)
print(dataset.head(10))

plt.xlabel('Area')
plt.ylabel('Price')

x = dataset.drop('price', axis='columns')
y = dataset.price

print(x.dtypes)
print(y.dtypes)
plt.scatter(x, y, marker='*', color='red')

model = LinearRegression()
model.fit(x, y)

a = [[int(input("Enter the land area (in square feet) :"))]]
predicted_value = model.predict(a)
print("Predicted value is :", predicted_value)

m = model.coef_
c = model.intercept_

print("Slope :", m )
print("Intercept :", c)
print("Actual Value :", m*a+c)
plt.plot(x, model.predict(x))
plt.show()
