import pandas as pd
# import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

dataset = pd.read_csv('data.csv')
print(dataset.shape)
# print(dataset.head(20))

dataset['diagnosis'] = dataset['diagnosis'].map({'M':1, 'B':0}).astype(int)
# print(dataset.head(20))

x = dataset.iloc[:, 2:32]
y = dataset.iloc[:, 1]

# print(x)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

sc= StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

models =[]

models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('RF', RandomForestClassifier()))

results=[]
names=[]
res=[]

for name, model in models:
    k_fold = StratifiedKFold(n_splits=10, random_state=None)
    cv_results = cross_val_score(model, x_train, y_train, cv=k_fold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    res.append(cv_results.mean()*100.0)
    print('%s : %f(%f)' % (name, cv_results.mean()*100.0, cv_results.std()))

plt.ylim(90.0, 99.9)
plt.bar(names, res, color='maroon', width=0.6)
plt.title('Algorithm comparison')
plt.show()
