import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('fetal_health.csv')

print(dataset.shape)
print(dataset.head(10))

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

model = LogisticRegression(solver='liblinear')
model.fit(x_train, y_train)

print("x_test", x_test.shape)
print("y_test", y_test.shape)
# plt.scatter(x_test, y_test, marker='*')

y_pred = model.predict(x_test)
# plt.plot(x_test, y_pred)
# plt.show()

print("Accuracy :{0}".format(accuracy_score(y_test, y_pred)* 100))
