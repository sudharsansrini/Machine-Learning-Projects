import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

dataset = load_iris()

print(dataset.data.shape)
print(dataset.target.shape)
#print(dataset)
x = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = dataset.target

# print(x)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
accuracy = []

for i in range(1, 10):
    model = DecisionTreeClassifier(max_depth=i, random_state=0)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = accuracy_score(y_test, y_pred)*100
    accuracy.append(score)

plt.figure(figsize=(12, 6))
plt.plot(range(1, 10), accuracy, color="red", linestyle="dashed", marker="o", markerfacecolor="blue", markersize=10)
plt.title('Finding the max depth')
plt.xlabel('pred')
plt.ylabel('score')
plt.show()

model = DecisionTreeClassifier(max_depth=3.5, random_state=0, criterion='entropy')
model.fit(x_train, y_train)

sepal_length = float(input("Enter Sepal length :"))
sepal_width = float(input("Enter Sepal Width :"))
petal_length = float(input("Enter petal length :"))
petal_width = float(input("Enter petal width :"))

iris = [[sepal_length, sepal_width, petal_length, petal_width]]
result = model.predict(iris)
if result == 0:
    print(dataset.target_names[0])
elif result == 1:
    print(dataset.target_names[1])
else:
    print(dataset.target_names[2])

y_pred = model.predict(x_test)
# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

print("Accuracy Score : {0}%".format(accuracy_score(y_test, y_pred)*100))