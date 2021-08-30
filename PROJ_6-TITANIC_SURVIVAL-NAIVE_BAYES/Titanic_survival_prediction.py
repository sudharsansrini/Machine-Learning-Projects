import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB

from sklearn.metrics import accuracy_score

dataset = pd.read_csv('titanicsurvival.csv')
# print(dataset.shape)
# print(dataset.head(10))

dataset['Sex'] = dataset['Sex'].map({'male': 1, 'female': 0}).astype(int)

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
# print(x)
# print(y)


# print(x.isna().any())
x.Age = x.Age.fillna(x.Age.mean())



# print(x.columns[x.isna().any()])
# print(x.head(10))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

model = GaussianNB()
model1 = MultinomialNB()
model2 = ComplementNB()
model3 = BernoulliNB()
model4 = CategoricalNB()

model.fit(x_train, y_train)
model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)
model4.fit(x_train, y_train)


p_class = int(input("Enter p class :"))
gender = int(input("Enter your gender (0: female, 1: male) :"))
age = int(input("Enter your age :"))
fare = int(input("Enter fare :"))

person=[[p_class, gender, age, fare]]

result = model.predict(person)
print(result)

if result == 0:
    print("Theh person might not survived")
else:
    print("The person might be survived")

y_pred = model.predict(x_test)
y_pred_1 = model1.predict(x_test)
y_pred_2 = model2.predict(x_test)
y_pred_3 = model3.predict(x_test)
y_pred_4 = model4.predict(x_test)



y_pred = np.array(y_pred)
print(y_pred.shape)

# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

print("Accuracy (Gaussian): {0}%".format(accuracy_score(y_test, y_pred)*100))
print("Accuracy (Multinomial) : {0}%".format(accuracy_score(y_test, y_pred_1)*100))
print("Accuracy (Complement) : {0}%".format(accuracy_score(y_test, y_pred_2)*100))
print("Accuracy (Bernoulli) : {0}%".format(accuracy_score(y_test, y_pred_3)*100))
print("Accuracy (Categorical) : {0}%".format(accuracy_score(y_test, y_pred_4)*100))



