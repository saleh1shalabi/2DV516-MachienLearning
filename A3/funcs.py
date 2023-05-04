import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from threading import Thread


def fitter_maker(C, g, d):

  """
  makes all combinations of estimators
  used since it was faster than gridsearchCv
  Args:
    C:
    g:
    d:

  Returns:

  """

  linears = []
  clfs = []
  polys = []
  for i in range(len(C)):
    cl = SVC(kernel="linear", C=C[i])
    linears.append(cl)
    for j in range(len(g)):
      cl = SVC(kernel="rbf", C=C[i], gamma=g[j])
      clfs.append(cl)
    for j in range(len(d)):
      cl = SVC(kernel="poly", C=C[i], degree=d[j])
      polys.append(cl)
  return linears, clfs, polys


def fitter_chooser(alist, X, y):

  """
  fitts a list of estimateros
  Args:
    alist:
    X:
    y:

  Returns:

  """

  scor = 0
  ind = 0
  for i in range(len(alist)):
    score =  alist[i].score(X, y)
    if scor < score:
      scor = score
      ind = i
  return ind, scor





def dec_boundry(grid_size, skl, X, trees = False):

  """
  decision noundery grid maker
  Args:
    grid_size:
    skl: estimator
    X: the X set
    trees: for the forest

  Returns:

  """
  min_value_x, max_value_x = min(X[:, 0]) - 0.5, max(X[:, 0]) + 0.5
  min_value_y, max_value_y = min(X[:, 1]) - 0.5, max(X[:, 1]) + 0.5

  x_axis = np.linspace(min_value_x, max_value_x, grid_size)
  y_axis = np.linspace(min_value_y, max_value_y, grid_size)

  xx, yy = np.meshgrid(x_axis, y_axis)
  cells = np.stack([xx.ravel(), yy.ravel()], axis=1)
  if trees:
    grid = np.zeros(shape=(len(x_axis), len(y_axis)))
    for t in skl:
      grid += t.predict(cells).reshape(grid_size,grid_size)
    zeros = np.where(grid < 50)
    ones = np.where(grid >= 50)
    grid[zeros] = 0
    grid[ones] = 1
  else:
    grid = skl.predict(cells).reshape(grid_size, grid_size)

  return grid











## threads is used to preform faster.
def estimators_fitter(alist, X, y):
  """
  uses multi threads to fit estimators faster
  Args:
    alist:
    X:
    y:

  Returns:

  """
  threads = []
  for c in alist:
    t = Thread(target=c.fit, args=(X, y))
    t.start()
    threads.append(t)

  for t in threads:
    t.join()


def points_and_grid_plot(grid, cmap_b, cmap_l, X, y):
  plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_b)
  plt.imshow(grid, origin='lower',
             extent=(min(X[:, 0]) - 0.5, max(X[:, 0]) + 0.5,
                     min(X[:, 1]) - 0.5, max(X[:, 1]) + 0.5), cmap=cmap_l)


def multi_to_bi(y):
  """
  used for the one vs all
  changes the values of y to 1 or 0
  Args:
    y:

  Returns:

  """
  sets = []
  for n in range(10):
    set = y == n
    sets.append(np.array(set).T)
  return sets


def one_vs_all_predict_proba(models, X):

  """
  predicts the outcome of the forest
  Args:
    models:
    X:

  Returns:

  """

  y_predict_one_vs_all = np.zeros(X.shape[0])
  for i in models:
    f = i.predict_proba(X)[:, 1]  # index of the probability
    y_predict_one_vs_all = np.column_stack((y_predict_one_vs_all, f))

  y_predict_one_vs_all = np.delete(y_predict_one_vs_all, 0, axis=1)  ## remove the zeroes from start
  y_predict_one_vs_all = np.argmax(y_predict_one_vs_all, axis=1)
  return y_predict_one_vs_all


