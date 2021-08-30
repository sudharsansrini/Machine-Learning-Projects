import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt

names=['voltage', 'current', 'power', 'class']
dataset = pd.read_csv('Energy Meter.csv', names = names)

array = dataset.values
x = array[:, :3]
y = array[:, 3]

# print(dataset.shape)
# print(dataset.head(20))
# print(dataset.describe())
# print(dataset.groupby('class').size())
# print(x)
# print(y)

x_train , x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.80, random_state=1)


model = SVC(gamma='auto')
model.fit(x_train, y_train)

result = model.score(x_validation, y_validation)
# print(result)

value = [[289.23, 0.9234, 182.333]]
predict = model.predict(value)
print(predict)

