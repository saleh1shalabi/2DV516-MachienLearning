import os
from os.path import isfile, join, isdir

import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from test import sammon
import matplotlib.pyplot as plt
# from MNIST_FileReader import get_train_set

def normalize_values(X):
  mean_values = []
  std_values = []
  for i in range(X.shape[1]):
      mean_values.append(np.mean(X[:, i]))
      std_values.append(np.std(X[:, i]))

  for i in std_values:
    if i < 1e-8 : i = 1e-5




    Xn = []
    for i in range(X.shape[1]):
      g = np.nan_to_num((X[:, i] - mean_values[i]) / std_values[i])
      Xn.append(np.array(g).T)
    Xn = np.array(Xn).T
    return Xn






########################################################################################
# https://www.kaggle.com/datasets/muratkokludataset/raisin-dataset
labels = np.loadtxt("raisin.csv", dtype=str, delimiter=",", max_rows=1)
labels = labels[:-1]
data = np.loadtxt("raisin.csv", dtype=str, delimiter=",", skiprows=1)
np.random.shuffle(data)
data = np.array_split(data, 9)[0]
X = data[:,:data.shape[1]-1]
X = X.astype(float)

y = data[:,data.shape[1]- 1]

uni = np.unique(y)

k=0
for i in uni:
  y[y == i] = k
  k+=1

y_raisin = y.astype(int)

X_raisin = normalize_values(X)
###############################################################################
labels = np.loadtxt("bean.csv", dtype=str, delimiter=",", max_rows=1)
labels = labels[:-1]
data = np.loadtxt("bean.csv", dtype=str, delimiter=",", skiprows=1)
np.random.shuffle(data)
data = np.array_split(data, 120)[0]




X = data[:,:data.shape[1]-1]
X = X.astype(float)

y = data[:,data.shape[1]- 1]

uni = np.unique(y)

k=0
for i in uni:
  y[y == i] = k
  k+=1

y_beans = y.astype(int)


X_beans = normalize_values(X)



###############################################################################


labels = np.loadtxt("mobile.csv", dtype=str, delimiter=",", max_rows=1)
labels = labels[:-1]
data = np.loadtxt("mobile.csv", delimiter=",", skiprows=1)
data = np.array_split(data, 16)[0]
X_mobile = data[:,:data.shape[1]-1]


y_mobile = data[:,data.shape[1]- 1]
y_mobile = y_mobile.astype(np.int32)
uni = np.unique(y)

# X_mobile = normalize_values(X_mobile)
############################################################################
print(X_raisin.shape)
print(X_beans.shape)
print(X_mobile.shape)

print(np.unique(y_raisin))
print(np.unique(y_beans))
print(np.unique(y_mobile))


print("PCA")
pca = PCA(n_components=2)

pca_raisin = pca.fit_transform(X_raisin)
pca_beans = pca.fit_transform(X_beans)
pca_mobile = pca.fit_transform(X_mobile)


tsne = TSNE(n_components=2, init="random", learning_rate="auto")

print("T-SNE")
tsne.fit(X)
tsne_raisin = tsne.fit_transform(X_raisin)
tsne_beans = tsne.fit_transform(X_beans)
tsne_mobile = tsne.fit_transform(X_mobile)



plt.figure(figsize=(15,15))
print("Sammon")


#
# sammon_raisin, col_raisin= sammon(X_raisin, alpha=0.0009, iter=5000, error=0.02, chunk_size=25)
# sammon_beans, col_beans= sammon(X_beans, alpha=0.0009, iter=5000, error=0.02, chunk_size=25)
# sammon_mobile, col_mobile= sammon(X_mobile, alpha=0.0009, iter=5000, error=0.02, chunk_size=25)
#
#
#
#
# plt.subplot(3,3,1)
# plt.title("Pca raisin")
# plt.scatter(pca_raisin[:,0], pca_raisin[:,1], c = y_raisin)
# plt.subplot(3,3,4)
# plt.title("Pca beans")
# plt.scatter(pca_beans[:,0], pca_beans[:,1], c = y_beans)
# plt.subplot(3,3,7)
# plt.title("Pca mobile")
# plt.scatter(pca_mobile[:,0], pca_mobile[:,1], c = y_mobile)
#
#
#
#
# plt.subplot(3,3,2)
# plt.title("tsne raisin")
# plt.scatter(tsne_raisin[:,0], tsne_raisin[:,1], c = y_raisin)
# plt.subplot(3,3,5)
# plt.title("tsne beans")
# plt.scatter(tsne_beans[:,0], tsne_beans[:,1], c = y_beans)
# plt.subplot(3,3,8)
# plt.title("tsne mobile")
# plt.scatter(tsne_mobile[:,0], tsne_mobile[:,1], c = y_mobile)
#
#
# plt.subplot(3,3,3)
# plt.title("sammon raisin")
# plt.scatter(sammon_raisin[:,0], sammon_raisin[:,1], c = y_raisin)
# plt.subplot(3,3,6)
# plt.title("sammon beans")
# plt.scatter(sammon_beans[:,0], sammon_beans[:,1], c = y_beans)
# plt.subplot(3,3,9)
# plt.title("sammon mobile")
# plt.scatter(sammon_mobile[:,0], sammon_mobile[:,1], c = y_mobile)
#



# plt.show()
# exit()

from ss import bkmeans
from sklearn.cluster import KMeans, AgglomerativeClustering

bk_raisin = bkmeans(tsne_raisin, 2)
bk_beans = bkmeans(tsne_beans, 7)
bk_mobile = bkmeans(tsne_mobile, 4)


k_raisin = KMeans(n_clusters=2).fit_predict(tsne_raisin)
k_beans = KMeans(n_clusters=7).fit_predict(tsne_beans)
k_mobile = KMeans(n_clusters=4).fit_predict(tsne_mobile)

agg_raisin = AgglomerativeClustering(n_clusters=2).fit_predict(tsne_raisin)
agg_beans = AgglomerativeClustering(n_clusters=7).fit_predict(tsne_beans)
agg_mobile = AgglomerativeClustering(n_clusters=4).fit_predict(tsne_mobile)





plt.subplot(3,3,1)
plt.title("bk raisin")
plt.scatter(tsne_raisin[:,0], tsne_raisin[:,1], c = bk_raisin, cmap="Set1")

plt.subplot(3,3,4)
plt.title("bk beans")
plt.scatter(tsne_beans[:,0], tsne_beans[:,1], c = bk_beans, cmap="Set1")
plt.subplot(3,3,7)
plt.title("bk mobile")
plt.scatter(tsne_mobile[:,0], tsne_mobile[:,1], c = bk_mobile, cmap="Set1")




plt.subplot(3,3,2)
plt.title("k raisin")
plt.scatter(tsne_raisin[:,0], tsne_raisin[:,1], c = k_raisin, cmap="Set1")
plt.subplot(3,3,5)
plt.title("k beans")
plt.scatter(tsne_beans[:,0], tsne_beans[:,1], c = k_beans, cmap="Set1")

plt.subplot(3,3,8)
plt.title("k mobile")
plt.scatter(tsne_mobile[:,0], tsne_mobile[:,1], c = k_mobile, cmap="Set1")


plt.subplot(3,3,3)
plt.title("agg raisin")
plt.scatter(tsne_raisin[:,0], tsne_raisin[:,1], c = agg_raisin, cmap="Set1")
plt.subplot(3,3,6)
plt.title("agg beans")
plt.scatter(tsne_beans[:,0], tsne_beans[:,1], c = agg_beans, cmap="Set1")

plt.subplot(3,3,9)
plt.title("agg mobile")
plt.scatter(tsne_mobile[:,0], tsne_mobile[:,1], c = agg_mobile, cmap="Set1")

plt.show()
