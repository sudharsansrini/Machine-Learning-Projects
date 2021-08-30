import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dataset = pd.read_csv('dataset.csv')
# print(dataset.head(5))
print(dataset.shape)
print(dataset.describe())

income = dataset['INCOME']
spent = dataset['SPEND']

x = np.array(list(zip(income, spent)))
print(x.shape)

plt.scatter(x[:, 0], x[:, 1])
plt.show()

wcss =[]
for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(x)
    wcss.append(km.inertia_)

plt.plot(range(1, 11), wcss, color='red', marker='8')
plt.title('Choosing K value')
plt.xlabel('No of clusters')
plt.ylabel('WCSS')
plt.show()

model = KMeans(n_clusters=6, random_state=0)
y_means = model.fit_predict(x)
print(y_means)

plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], c='brown', s=50, label='low INC low SPT')
plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], c='blue', s=50, label='low INC high SPT')
plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], c='green', s=50, label='high INC')
plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], c='cyan', s=50, label='avg INC high SPT')
plt.scatter(x[y_means == 4, 0], x[y_means == 4, 1], c='pink', s=50, label='low INC high SPT')
plt.scatter(x[y_means == 5, 0], x[y_means == 5, 1], c='violet', s=50, label='avg INC low SPT')
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='red', s=70, label='centroids')
plt.title('INCOME ANALYSIS')
plt.xlabel('INCOME')
plt.ylabel('SPENT')
plt.legend()
plt.show()


