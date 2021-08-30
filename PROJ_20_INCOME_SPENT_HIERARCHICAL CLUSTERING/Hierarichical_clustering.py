import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import scipy.cluster.hierarchy as clus
from sklearn.cluster import AgglomerativeClustering

dataset = pd.read_csv('dataset.csv')
print(dataset.shape)
# print(dataset.head(5))
# print(dataset.describe())

dataset = dataset.drop('CustomerID', axis='columns')
# print(dataset.head(5))

label_encoder = LabelEncoder()
dataset['Gender'] = label_encoder.fit_transform(dataset['Gender'])
print(dataset.head(5))

"""
plt.hist([dataset.iloc[:, 2], dataset.iloc[:, 3]])
plt.show()
"""

plt.scatter(dataset.iloc[:, 2], dataset.iloc[:, 3])
plt.show()

x = dataset.iloc[:, [2, 3]].values

plt.figure(figsize=(10,10))
dendrogram = clus.dendrogram(clus.linkage(dataset, method="ward"))
plt.title('Dendrogram Tree Graph')
plt.xlabel('Customer')
plt.ylabel('Distances')
plt.show()
model = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='average')
y_means = model.fit_predict(dataset)
print(y_means)

plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s=50, c='blue', label='Cluster 1')
plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s=50, c='orange', label='Cluster 2')
plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s=50, c='purple', label='Cluster 3')
plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], s=50, c='green', label='Cluster 4')
plt.scatter(x[y_means == 4, 0], x[y_means == 4, 1], s=50, c='yellow', label='Cluster 5')
plt.scatter(x[y_means == 5, 0], x[y_means == 5, 1], s=50, c='cyan', label='Cluster 6')
plt.title('INCOME SPENT - HIERARCHICAL CLUSTERING')
plt.xlabel('INCOME')
plt.ylabel('SPENT')
plt.show()

