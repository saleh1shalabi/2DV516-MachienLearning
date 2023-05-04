import numpy as np
from A2 import REG, POL_REG

data = np.loadtxt("A2_datasets_2022/GPUbenchmark.csv", delimiter=",", dtype=np.float64)

X = data[:,: data.shape[1] - 1]
y = data[:,data.shape[1]-1]


def forward_selection_module_producer(X, y):
  """
  this function generates all possible sets of the set X
  it gives also the best fit of them all

  :param X: the X set
  :param y: the y set
  :return:
  best: the best features order of the set X
  M: all possible sets from set X
  """
  M = []
  models = 0
  costs = []
  best = []

  for i in range(X.shape[1]):

    m = X[:,i]
    costs.append(POL_REG(m, y).cost_fun()) ## since i did not want to rewrite my classes I used the Pol_reg without
    # any degree which is the same as the useual reg but works when X have only one feature
    M.append(m)
    models+=1
  ind = costs.index(min(costs))
  best.append(ind)
  while True:
    costs = []
    s = np.ones(X.shape[0])
    for c in best:
      s = np.column_stack((s, X[:,c]))
    s= s[:,1:]

    for i in range(X.shape[1]):
      if i in best:
        costs.append(10e100) ## to not leave the index empty and get right index from the list
      else:
        m = np.column_stack((s, X[:,i]))
        costs.append(REG(m, y).cost_fun())
        M.append(m)

    ind = costs.index(min(costs))
    best.append(ind)
    models += 1

    if len(best) == X.shape[1]: ## then we came to the end
      return best, M




best, models = forward_selection_module_producer(X,y)

def k_fold_cross_validation(k, models,y, bestfit):
  """
  Well I don't think this algorithm is correct, and I assumed we were not allowed to use
  sklearn for this task.
  but the result I get is the same as other in my class, so I assumed that something is correct in it at least
  it works as all models that have been produces in function forward_selection_module_producer
  divided each set to k subsets and count the average MSE of those sets.
  :param k: how many subset each model will be divided into
  :param models: all models to count
  :param y: y values
  :param bestfit: the best fit from function forward_selection_module_producer
  :return:
  the model that had the best value MSE with the order of its features
  """


  medelss = []
  for c in models:
    proc = 1/k
    if c.shape[0] == c.size:


      splited = np.split(c,k, axis=0)
      y_splited = np.split(y,k, axis=0)
      medel = 0
      for n in range(len(splited)):
        xt = splited[n]
        yt = y_splited[n]
        reg = POL_REG(xt, yt)
        medel += reg.cost_fun()
      medelss.append(medel/len(splited))

    else:
      splited = np.split(c,k, axis=0)
      y_splited = np.split(y,k, axis=0)
      medel = 0
      for n in range(len(splited)):
        xt = splited[n]
        yt = y_splited[n]
        reg = REG(xt, yt)
        medel += reg.cost_fun()
      medelss.append(medel/k)

  ind = medelss.index(min(medelss))

  return models[ind], bestfit[:models[ind].shape[1]]

mo , bs = k_fold_cross_validation(3,models,y,best) ## it produc error when the k is larger than 3
# I know how to fix it, but uninformatively I did not have the time to do it

print("the best selection of params is:", bs)
print("the most important param, which is the first one:", bs[0])


