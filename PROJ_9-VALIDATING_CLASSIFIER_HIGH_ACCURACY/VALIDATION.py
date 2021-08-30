# 1 - CONFUSION MATRIX
# 2 - ROC CURVE
# 3 - CROSS VALIDATION SCORE
# 4 - STRATIFIED K-FOLD CROSS VALIDATION
# 5 - CUMMULATIVE ACCURACY PROFILE (CAP)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('DigitalAd_dataset.csv')

# print(dataset.shape)
# print(dataset.head(5))

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

model = LogisticRegression(random_state=0)
model.fit(x_train, y_train)

"""
age = int(input("Customers Age :"))
sal = int(input("Customers Salary :"))

newCust = [[age, sal]]
result = model.predict(sc.transform(newCust))
print(result)

if result == 1:
    print("Customer will buy")
else:
    print("Customer won't buy")
"""
y_pred = model.predict(x_test)
# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))


# 1 - CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
# print(cm)

print("CONFUSION MATRIX Accuracy : {0}%".format(accuracy_score(y_pred, y_test)*100))

# 2 - ROC CURVE

from sklearn.metrics import roc_auc_score, roc_curve

nsProb =[0 for i in range(len(y_test))]
# print(nsProb)
lsProb = model.predict_proba(x_test)
lsProb = lsProb[:, 1]
# print(lsProb)

# calculating scores

nsAUC = roc_auc_score(y_test, nsProb)
lrAUC = roc_auc_score(y_test, lsProb)
# print(nsAUC)
# print(lrAUC)

# summarize scores

# print('No skill : ROC CURVE = %.2f'% (nsAUC*100))
print('Logistic Regression skill : ROC CURVE = %.2f'% (lrAUC* 100))

# calculating ROC curve

nsFP, nsTP, _ = roc_curve(y_test, nsProb)
lrFP, lrTP, _ = roc_curve(y_test, lsProb)

# print(lrFP)
# print(lrTP)
# Plot the roc curve for the model

plt.plot(nsFP, nsTP, linestyle="dashed", label="No skill")
plt.plot(lrFP, lrTP, marker="o", markersize=2, color="red",  label="logistic")
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend()
plt.show()

# 3. cross validation

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

kfold = KFold(n_splits=10, random_state=100, shuffle=False)
result = cross_val_score(model, x, y, cv=kfold)
print("Cross validation Score = %.2f%%"% (result.mean()*100.0))

# 4. STRATIFIED K-FOLD
from sklearn.model_selection import StratifiedKFold

sk_fold = StratifiedKFold(n_splits=3, random_state=100)
result_skfold = cross_val_score(model, x, y, cv=sk_fold)

print("Stratified k-fold score =%.2f%%" %(result_skfold.mean()*100))

# 5 - CUMMULATIVE ACCURACY PROFILE

total = len(y_test)
print(total)

class_1_count = np.sum(y_test)
print(class_1_count)

class_0_count = total - class_1_count
print(class_0_count)

plt.plot([0, total], [0, class_1_count], c='r', linestyle="--", label='random model')
plt.plot([0, class_1_count, total], [0, class_1_count, class_1_count], linewidth=2, label="perfect model")

probs = model.predict_proba(x_test)
probs = probs[:, 1]

model_y = [y for _, y in sorted(zip(probs, y_test), reverse=True)]
# print(model_y)
y_values = np.append([0], np.cumsum(model_y))
x_values = np.arange(0, total+1)
print(x_values)
print(y_values)

plt.plot(x_values, y_values, c='b', label='LR classifier', linewidth=4)

index = int((50* total/100))
# 50% vertical line from y axis
plt.plot([index, index], [0, y_values[index]], c='g', linestyle="dashed")

# horizontal line to x axis
plt.plot([0, index], [y_values[index], y_values[index]], c='g', linestyle="dashed")

class_1_observed = y_values[index] * 100/max(y_values)
plt.xlabel('Total observations')
plt.ylabel('Class 1 observations')
plt.title('Cummulative Accuracy Profile')
plt.legend(loc='lower right')

plt.show()
