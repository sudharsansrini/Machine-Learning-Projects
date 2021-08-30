import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

dataset = pd.read_csv('salary.csv')
print(dataset.shape)
# print(dataset.head(5))

dataset['income'] = dataset['income'].map({'<=50K': 0, '>50K': 1}).astype(int)
# print(dataset.head(10))

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# print(x)

# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# print(x_train)
# print(x_test)

error =[]
""" USED FOR CHOOSING K VALUE - VERY VERY IMPORTANT STEP  
for i in range(1, 40):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(x_train, y_train)
    pred_i = model.predict(x_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(15, 15))
plt.plot(range(1, 40), error, color = "red", linestyle = "dashed", marker="o", markerfacecolor="blue", markersize=10)
plt.title('Error Rate K value')
plt.xlabel('K value')
plt.ylabel('Mean Error')
plt.show()
"""

model = KNeighborsClassifier(n_neighbors=16 , metric='minkowski', p=2)
model.fit(x_train, y_train)

age = int(input("Enter Age :"))
edu = int(input("Enter Education :"))
cg = int(input("Enter capital gain :"))
wh = int(input("Enter hours/ week :"))

newEmp = [[age, edu, cg, wh]]

result = model.predict(sc.transform(newEmp))
if result == 0 :
    print("Employee Salary Might not above 50K")
else:
    print("Employee Salary Might above 50k")

y_pred = model.predict(x_test)
# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix :", cm)
print("Accuracy value : {0}%".format(accuracy_score(y_test, y_pred)*100))




