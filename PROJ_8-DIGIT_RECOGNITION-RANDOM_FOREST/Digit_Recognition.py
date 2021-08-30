import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

dataset = pd.read_csv('digit.csv')
# print(dataset.shape)

x = dataset.iloc[:, 1:]
y = dataset.iloc[:, 0]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

model = RandomForestClassifier()
model.fit(x_train, y_train)
index= 123
print("Predicted :" + str(model.predict(x_test)[index]))
plt.axis('off')
plt.imshow(x_test.iloc[index].values.reshape((28, 28)), cmap='gray')
plt.show()

y_pred = model.predict(x_test)
print("Accuracy :{0}%".format(accuracy_score(y_test, y_pred)*100))

