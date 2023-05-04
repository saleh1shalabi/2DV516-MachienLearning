import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs, make_s_curve, make_swiss_roll
from threading import Thread

# np.seterr(all="ignore")

def euclidean_distance(X, point):

    return np.linalg.norm(X - point, axis=1)


def array_distances(X):
  return pdist(X, "euclidean") # faster than my implementation :(

  # good when X sampels not larger than 1000
  # dists = np.array([])
  # i = 1
  # for i in range(X.shape[0]):
  #   j = X[i+1:]
  #   z = np.full(j.shape, X[i])
  #   o = euclidean_distance(z, j)
  #   dists = np.append(dists, o)
  #   i+=1
  # return dists




def sammon_stress(d_dists, real_dists):
  d_dists[d_dists < 1e-10] = 1e-7
  real_dists[real_dists < 1e-10] = 1e-7

  r = np.sum(real_dists)

  delta_dists = (d_dists - real_dists) ** 2
  diff = np.nan_to_num(delta_dists / real_dists)

  E = (1/r) * sum(diff)

  return E





def best_params(X):
  dists_X = array_distances(X)
  b_f = {}

  for i in range(X.shape[1]):
    l = {}
    for j in range(X.shape[1]):
      if i != j:
        y = X[:,[i,j]]
        dists_y = array_distances(y)
        l[j] = sammon_stress(dists_X, dists_y)

    b_f[i] = l

  least = []
  for k, v in b_f.items():
    least.append(min(v.values()))

  # print(min(least))

  params = []

  for k, v in b_f.items():
    for i, j in v.items():
      if j == min(least):
        params.append(k)
        params.append(i)


  return list(set(params))





# best_params(make_blobs(n_samples=100, n_features=15, centers=12,random_state=4 )[0])
# exit()






def gradient(X, y , y_new, i, alpha, c):
  Xj = np.delete(np.copy(X), i, axis=0)
  yj = np.delete(np.copy(y), i, axis=0)
  Xi = np.full(Xj.shape, X[i])
  yi = np.full(yj.shape, y[i])

  xij_dists = euclidean_distance(Xi, Xj).reshape(Xi.shape[0], 1)

  xij_dists[xij_dists< 1e-10] = 1e-10

  yij_dists = euclidean_distance(yi, yj).reshape(Xi.shape[0], 1)  # reshape
  yij_dists[yij_dists< 1e-10] = 1e-10

  yi_diff_yj = yi - yj

  xy_diff = xij_dists - yij_dists

  denom = xij_dists * yij_dists
  denom[denom < 1e-10] = 1e-10

  p1 = np.nan_to_num(-2/c) * np.sum(np.nan_to_num(xy_diff/denom) * yi_diff_yj)

  t = 1 + np.nan_to_num(xy_diff / yij_dists)

  dd = np.nan_to_num( (yi_diff_yj ** 2) / yij_dists )
  p2 = np.nan_to_num(-2/c) * np.sum(np.nan_to_num(1/denom) * (xy_diff - (dd*t)))
  delta = p1 / np.abs(p2)
  y_new[i] = y[i] - alpha * delta

def sammon(X, alpha, iter, error, y_s = 2, chunk_size = 5, best_feat = False):
  columns = []

  if best_feat:
    columns = best_params(X)

  else:
    for i in range(y_s):
      rand_col = random.randint(0, X.shape[1]-1)
      while rand_col in columns:
        rand_col = random.randint(0, X.shape[1]-1)
      columns.append(rand_col)

  y = X[:, columns]
  dists_x = array_distances(X)
  print("array calcs")

  c = np.sum(dists_x)
  y_new = np.copy(y)
  E_old = 100 # just a start value
  delta_old = 10
  for n in range(iter):

    if n == 0:
      dists_y = array_distances(y)
      E_old = sammon_stress(dists_y, dists_x)
      E = E_old

    else:
      dists_y = array_distances(y_new)
      E = sammon_stress(dists_y, dists_x)

    if E < error or ( abs(E_old - E) > error*1.5 and -(E_old - E) > (E_old - E)):
      break

    else:
      y = np.copy(y_new)
      E_old = E

    print("iter:", n, "E:", E)

    ts = []
    for i in range(X.shape[0]):
      if len(ts) != chunk_size:
        t = Thread(target=gradient, args=(X, y, y_new, i, alpha, c))
        t.start()
        ts.append(t)
      else:
        for t in ts:
          t.join()
        # print(i)
        ts = []



    # y = np.array(y_new)
    # print(delta_old, delta)
    # if abs(delta_old) < abs(delta) : ## if the gradient starts to go up we should break
    #   # print("delta break")
    #   break
    # else:
    #   delta_old = delta
  print(E if E < E_old else E_old)
  return y, columns







def plot_for_sammon(data):

  X = data[0]
  y = data[1]


  fig = plt.figure()
  ax = plt.axes(projection='3d')


  x_ax = X[:,0]
  y_ax = X[:,1]
  z_ax = X[:,2]

  ax.scatter3D(x_ax, y_ax, z_ax, c = y)

  x , xx= sammon(X, alpha=0.005, iter=100, error=0.02)

  print(x.shape)
  plt.figure()
  plt.scatter(x[:,0], x[:,1], c = y)
  plt.show()




def main():

  # roll = make_swiss_roll(n_samples=100, noise=0.15, random_state=7)
  # blobs = make_blobs(n_samples=10000, n_features=15, centers=12,random_state=4 )
  s_curve = make_s_curve(n_samples=1000, noise= 0.05, random_state=7)
  # plot_for_sammon(roll)
  plot_for_sammon(s_curve)
  # plot_for_sammon(blobs)

# main()




