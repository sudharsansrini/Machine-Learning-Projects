import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


dataset = pd.read_csv('heart.csv')
print(dataset.shape)
# print(dataset.head(10))

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# print(x.shape)
# print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

print(x_train.shape)
print(x_test.shape)
"""
errors =[]
for i in range(1, 40):
     model = KNeighborsClassifier(n_neighbors=i)
     model.fit(x_train, y_train)
     y_pred = model.predict(x_test)
     errors.append(np.mean(y_pred != y_test))


plt.plot(range(1, 40), errors)
plt.show()
"""

model1 = KNeighborsClassifier(n_neighbors=14, metric='minkowski', p=2)
model2 = LogisticRegression(solver='liblinear')
model3 = SVC(kernel='linear', C=0.60, random_state=0)
model4 = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=0)
model5 = RandomForestClassifier()


model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)
model4.fit(x_train, y_train)
model5.fit(x_train, y_train)
"""
pred =[[67, 1, 0, 160, 286, 0 , 0, 108, 1, 1.5,	1, 3, 2]]

result = model1.predict(pred)
print(result)
"""

y_pred1 = model1.predict(x_test)
y_pred2 = model2.predict(x_test)
y_pred3 = model3.predict(x_test)
y_pred4 = model4.predict(x_test)
y_pred5 = model5.predict(x_test)

print("Accuracy - KNN :{0}".format(accuracy_score(y_pred1, y_test)*100))
print("Accuracy - LR :{0}".format(accuracy_score(y_pred2, y_test)*100))
print("Accuracy - SVM :{0}".format(accuracy_score(y_pred3, y_test)*100))
print("Accuracy - DTree :{0}".format(accuracy_score(y_pred4, y_test)*100))
print("Accuracy - RForest :{0}".format(accuracy_score(y_pred5, y_test)*100))

