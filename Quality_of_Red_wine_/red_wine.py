import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv('winequality-red.csv')
print(dataset.shape)
# print(dataset.head(10))


x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
print(x.shape)
print(y.shape)

for i in range(len(y)):
    if y[i] <= 6:
        y[i] = 0
    else:
        y[i] = 1


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

model = LogisticRegression(solver='liblinear')
model.fit(x_train, y_train)

pred = [[7.3, 0.65, 0, 1.2, 0.065, 15, 21, 0.9946, 3.39, 0.47, 10]]
result = model.predict(pred)
print(result)

y_pred = model.predict(x_test)

print("Accuracy :{0}".format(y_test, y_pred)* 100)