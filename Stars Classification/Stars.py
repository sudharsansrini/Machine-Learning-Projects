import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('Stars.csv')
print(dataset.shape)
# print(dataset.head(10))

# print(dataset.columns[dataset.isna().any()])

a = dataset['Color']
a1=[]

b= dataset['Spectral_Class']
b1=[]
for i in a:
    if i not in a1:
        a1.append(i)
for j in b:
    if j not in b1:
        b1.append(j)

print(a1)
print(b1)

x = dataset.iloc[:, :-3]
y = dataset.iloc[:, -1]
print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

sc= StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

accuracy =[]
for i in range(1, 20):
    model = DecisionTreeClassifier(max_depth=i, random_state=0)
    model.fit(x_train, y_train)
    y_pred_i = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred_i)*100
    accuracy.append(acc)

plt.figure(figsize=(20, 20))
# plt.ylim(90.0, 99,9)
plt.plot(range(1, 20), accuracy, marker='*', color="red", linestyle="--", markerfacecolor="blue", markersize=10)
plt.show()

model1 = DecisionTreeClassifier(max_depth=4, random_state=0)
model1.fit(x_train, y_train)

y_pred = model.predict(x_test)

plt.plot(y_test, y_pred, marker='o', color="green", linestyle="--", markerfacecolor="blue", markersize=10)
plt.show()
print("Accuracy :{0}".format(accuracy_score(y_pred, y_test)*100))


