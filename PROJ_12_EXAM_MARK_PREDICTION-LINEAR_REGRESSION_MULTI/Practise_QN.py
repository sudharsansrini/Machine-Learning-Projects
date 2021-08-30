import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv('practiceDataset.csv')
# print(dataset.shape)
# print(dataset.head(10))



# Mapping values - Find
s = []
s1 = []
sample = dataset['MSZoning']
sample1 = dataset['HouseStyle']

# print(sample)

for i in sample:
    if i not in s:
        s.append(i)

for j in sample1:
    if j not in s1:
        s1.append(j)
print(s)
print(s1)

dataset['MSZoning'] = dataset['MSZoning'].map({'RL':0, 'RM':1, 'C (all)':2, 'FV':3, 'RH':4 }).astype(int)
dataset['HouseStyle'] = dataset['HouseStyle'].map({'2Story':0, '1Story':1, '1.5Fin':2, '1.5Unf':3, 'SFoyer':4,
                                                   'SLvl':5, '2.5Unf':6, '2.5Fin':7}).astype(int)

x = dataset[['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'HouseStyle', 'OverallQual', 'OverallCond']]
print(x.columns[x.isna().any()])
x.LotFrontage = x.LotFrontage.fillna(x.LotFrontage.mean())


y = dataset.SalePrice

print(x)
"""
print(x.shape)
print(y.shape)
print(x.head(10))
"""
model = LinearRegression()
model.fit(x, y)

pred =[[20, 0, 80.0, 9600, 0, 7, 5]]
result = model.predict(pred)
print(result)
print("score :", model.score(x, y)*100)



