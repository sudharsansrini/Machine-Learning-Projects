import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

dataset = pd.read_csv('landpriceprediction.csv')

# print(dataset.shape)
# print(dataset.describe())
# print(dataset.groupby('price').size())

land = dataset.drop('price', axis='columns')
price = dataset.price

plt.scatter(land, price, marker='*', color='red')

model = linear_model.LinearRegression()
model.fit(land, price)

LandArea = [[5500]]

predicted = model.predict(LandArea)

print('Ans:', predicted)
print('Co-eff', model.coef_)
print('Intercept', model.intercept_)
plt.plot(land, model.predict(land))


plt.show()