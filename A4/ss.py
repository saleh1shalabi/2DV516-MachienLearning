from sklearn.cluster import KMeans
import numpy as np
from MNIST_FileReader import get_train_set
from sklearn.datasets._samples_generator import make_blobs
import matplotlib.pyplot as plt


# data = get_train_set(1000)
#
# X = data[:,:data.shape[1]-1]
# y = data[:,data.shape[1]-1]



def find_cent(X):
    s = np.sum(X, axis=0)
    s = s / X.shape[0]
    return s

def euclidean_distance(X, point):
    return np.linalg.norm(X - point)

def sse(X, point):
    return euclidean_distance(X, point) ** 2

def kmeans(X, iter, labels = None):
    SSEs = []

    y = np.zeros(X.shape[0], dtype=np.int32)
    ones = np.random.choice(X.shape[0], round(X.shape[0]/2), replace=False)
    y[ones] = 1
    zeros = np.where(y == 0)

    centroids = [find_cent(X[zeros]), find_cent(X[ones])]

    g = sse(X[zeros],centroids[0]) + sse(X[ones], centroids[1])
    SSEs.append(g)

    for i in range(iter):

        for j in range(X.shape[0]):
            if euclidean_distance(X[j], centroids[0]) < euclidean_distance(X[j], centroids[1]):
                y[j] = 0
            elif euclidean_distance(X[j], centroids[0]) > euclidean_distance(X[j], centroids[1]):
                y[j] = 1
            else: pass

        ## some maight not cahnge so we should look up all
        ones = np.where(y == 1)
        zeros = np.where(y == 0)

        centroids = [find_cent(X[zeros]), find_cent(X[ones])]

        g = sse(X[zeros], centroids[0])
        g += sse(X[ones], centroids[1])

        if g == SSEs[len(SSEs) - 1] and g == SSEs[len(SSEs) - 2]:
            break
        else: SSEs.append(g)

    if labels != None and len(labels) == 2:
        zeros = np.where(y == 0)
        ones = np.where(y == 1)
        y[zeros] = labels[0]
        y[ones] = labels[1]
    return y

def bkmeans(X, K, all = False):
    if K == 0:
        raise ValueError("K must be larger than 0")
    if K == 1:
        return np.zeros(X.shape[0])
    st = []
    y = np.zeros(X.shape[0])
    st.append(y)


    ## since we know that all is one value we run the first iter outside for efficency
    y = kmeans(X, 10, [0, 1])
    n = np.copy(y)
    st.append(n)

    clusters = 2

    while clusters != K:
        sums = []
        vals = []

        for i in np.unique(y):

            sums.append(np.sum(np.where(y == i)))
            vals.append(i)

        s = max(sums)
        s = sums.index(s)

        y_to_bi = np.where(y == vals[s])
        y[y_to_bi] = kmeans(X[y_to_bi], 100, [y[y_to_bi][0], clusters])
        n = np.copy(y)
        st.append(n)
        clusters += 1


    if all: return st
    return y


def main():

    data = make_blobs(n_samples=1000, n_features=2, random_state=0, centers=4)
    X = data[0]
    y = bkmeans(X,4)
    plt.scatter(X[:,0], X[:,1], c = y)
    plt.show()
