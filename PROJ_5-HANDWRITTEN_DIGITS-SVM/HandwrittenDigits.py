import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset = load_digits()
dataimageLenth = len(dataset.images)
# print(dataset.data)
# print(dataset.images)
# print(dataset)
# print(dataset.data.shape)
print(dataset.images.shape)

"""
n = 180        # testing process - put any for 'n' between 0 to 1797
plt.matshow(dataset.images[n])
plt.gray()
plt.show()
"""

x = dataset.images.reshape((dataimageLenth, -1))
# print(x)

y = dataset.target
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

model = svm.SVC(kernel='linear')
model.fit(x_train, y_train)

# checking with other kernels for greater accuracy
# 2. Radial Basis Function (Default)
model1 = svm.SVC(kernel='rbf')
model1.fit(x_train, y_train)


# By giving Gamma & c value in 'rbf'
model2 = svm.SVC(gamma=0.001, C=0.70)
model2.fit(x_train, y_train)

# predict

"""
n = int(input("Enter any values between 0 to 1797 :"))

result = model.predict(dataset.images[n].reshape(1, -1))
plt.imshow(dataset.images[n], cmap=plt.cm.gray_r, interpolation='nearest')
print(result)
print("\n")
plt.axis('off')
plt.title('%i'% result)
plt.show()
"""

y_pred = model.predict(x_test)
y_pred_1 = model1.predict(x_test)
y_pred_2 = model2.predict(x_test)
# print(y_pred.shape)
# print(y_test.shape)
# y_pred = np.array(y_pred)

# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Accuracy

print("Accuray score :{0}%".format(accuracy_score(y_test, y_pred) * 100))
print("Accuray score :{0}%".format(accuracy_score(y_test, y_pred_1) * 100))
print("Accuray score :{0}%".format(accuracy_score(y_test, y_pred_2) * 100))

