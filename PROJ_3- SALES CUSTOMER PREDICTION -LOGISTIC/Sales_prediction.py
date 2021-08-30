import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

dataset = pd.read_csv('DigitalAd_dataset.csv')

print(dataset.shape)
print(dataset.head(5))

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

x_train, x_test , y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# print(x_test)
# print(y_test)

# print(x_train)
# print(y_train)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
# print(x_train)
# print(x_test)

model = LogisticRegression(random_state=0, solver='liblinear')
model.fit(x_train, y_train)

age = int(input("Customers Age :"))
sal = int(input("Customers Salary :"))

newCust = [[age, sal]]
result = model.predict(sc.transform(newCust))
print(result)

if result == 1:
    print("Customer will buy")
else:
    print("Customer won't buy")

y_pred = model.predict(x_test)
# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

print(model.score(x_test, y_test))
print("Decision function :", model.decision_function(x_test))
print("Desify function :", model.densify())
print("Parameter :", model.get_params(deep=True))
print("Sparsify :", model.sparsify())

cm = confusion_matrix(y_test, y_pred)
print(cm)

print("Accuracy : {0}%".format(accuracy_score(y_pred, y_test)*100))
