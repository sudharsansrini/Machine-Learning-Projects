from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

dataset = datasets.load_iris()

print(dataset.data.shape)
print(dataset.target.shape)

# print(dataset.data)
# print(dataset.target)

x = dataset.data
y = dataset.target
names = dataset.target_names

model = PCA(n_components=4)
y_means = model.fit_transform(x)

plt.figure(figsize=(10,10))
color=['red','blue','orange','green']

for colors, i, names in zip(color, range(4), names):
    plt.scatter(y_means[y == i, 0], y_means[y == i, 1], color=colors, lw=2, label=names[i])

plt.title('IRIS Clustering')
plt.show()