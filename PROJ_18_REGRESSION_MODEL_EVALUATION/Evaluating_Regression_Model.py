import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np

dataset = pd.read_csv('dataset.csv')
print(dataset.shape)
print(dataset.head(10))

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
y_array = np.array(y)
ysvm = y_array.reshape((len(y_array)), 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

x_train_svm, x_test_svm, y_train_svm, y_test_svm = train_test_split(x, ysvm, test_size=0.25, random_state=0)
sc_x = StandardScaler()
sc_y = StandardScaler()
x_train_svm = sc_x.fit_transform(x_train_svm)
y_train_svm = sc_y.fit_transform(y_train_svm)

modelLR = LinearRegression()

PLR = PolynomialFeatures(degree=4)
x_poly = PLR.fit_transform(x_train)
modelPR = LinearRegression()

modelSVR = SVR()
modelKNN = KNeighborsRegressor()
modelDT = DecisionTreeRegressor()
modelRF = RandomForestRegressor()

modelLR.fit(x_train, y_train)
modelPR.fit(x_poly, y_train)

modelSVR.fit(x_train_svm, y_train_svm)
modelKNN.fit(x_train, y_train)
modelDT.fit(x_train, y_train)
modelRF.fit(x_train, y_train)

modelLR_pred = modelLR.predict(x_test)
modelPR_pred = modelPR.predict(PLR.transform(x_test))
modelSVR_pred = sc_y.inverse_transform(modelSVR.predict(sc_x.fit_transform(x_test)))
modelKNN_pred = modelKNN.predict(x_test)
modelDT_pred = modelDT.predict(x_test)
modelRF_pred = modelRF.predict(x_test)

print("Accuracy LR :{0}".format(r2_score(y_test, modelLR_pred)))
print("Accuracy PR :{0}".format(r2_score(y_test, modelPR_pred)))
print("Accuracy SVR :{0}".format(r2_score(y_test, modelSVR_pred)))
print("Accuracy KNN :{0}".format(r2_score(y_test, modelKNN_pred)))
print("Accuracy DT :{0}".format(r2_score(y_test, modelDT_pred)))
print("Accuracy RF :{0}".format(r2_score(y_test, modelRF_pred)))




