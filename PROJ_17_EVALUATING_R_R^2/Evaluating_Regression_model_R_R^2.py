import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('dataset1.csv')
print(dataset.shape)
print(dataset.head(10))

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

x_train, x_test , y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

model = LinearRegression()
model.fit(x, y)
plt.scatter(x_train, y_train, marker='*', color="red")
plt.plot(x, model.predict(x), linestyle="--", color="blue")
plt.show()

r2_score = model.score(x_test, y_test)

n = len(dataset)
p = len(dataset.columns)-1
print("n is {0} and p is {1}".format(n, p))
adjR = 1 - ((1-r2_score) * n-1/n-p-1)
print(r2_score)
print("Adjusted R2 :", adjR)