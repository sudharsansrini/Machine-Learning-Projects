import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt

dataset = pd.read_csv('dataset.csv')
print(dataset.shape)
# print(dataset.head(5))

# print(dataset.select_dtypes(['object']))

x = dataset.drop('price', axis='columns')

x = x.select_dtypes(exclude=['object'])
print(x.shape)

y = dataset.price

print(y.shape)

# print(scale(x))
# print(pd.DataFrame(scale(x)))

# print(x.columns)

cols = x.columns
x = pd.DataFrame(scale(x))
x.columns = cols

print(x.head(10))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)


accuracy = []
for i in range(1, 10):
    model = RandomForestRegressor(max_depth=i, random_state=0)
    model.fit(x_train, y_train)
    pred_i = model.predict(x_test)
    score = r2_score(y_test, pred_i)
    accuracy.append(score)

plt.plot(range(1, 10), accuracy, marker='*', color='red', linestyle='--', markerfacecolor='blue', markersize=10)
plt.show()


model1 = RandomForestRegressor()
model1.fit(x_train, y_train)

y_pred = model1.predict(x_test)

acc = r2_score(y_test, y_pred)*100
print("r2 score :", acc)
# print("Acc score :{0}%".format(accuracy_score(y_pred, y_test)* 100))
print("Model Score :{0}".format(model1.score(x_test, y_test)))
