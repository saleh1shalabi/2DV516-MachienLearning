import numpy as np


class REG:
  X = None
  y = None
  Xn = None
  Xe = None
  Xen = None
  mean_values = []
  std_values = []

  def __init__(self, arr, y):
    self.X = arr
    self.y = y
    ones = np.ones(self.X.shape[0])
    for i in range(arr.shape[1]):
      self.mean_values.append(np.mean(arr[:, i]))
      self.std_values.append(np.std(arr[:, i]))
    self.__normalize_values()
    self.Xe = np.column_stack((ones,self.X))
    self.Xen = np.column_stack((ones, self.Xn))


  def __normalize_values(self):
    self.Xn = []
    for i in range(self.X.shape[1]):
      g = (self.X[:, i] - self.mean_values[i]) / self.std_values[i]
      self.Xn.append(np.array(g).T)
    self.Xn = np.array(self.Xn).T



  def get_norm_set(self):
    """
    called to get the normalized dataset.
    :return:
      normalized data
    """
    return self.Xn

  def get_e_norm_set(self):
    """
    called to get the normalized extended dataset.
    :return:
      normalized extended data
    """
    return self.Xen

  def normal_equation(self, v=False):
    """
    function to get the beta from normal equation
    :param v: default False, to return beta of the non-normalized dataset: --> True
    :return:
    beta values of dataset
    """
    if v:
      return np.linalg.inv(self.Xe.T.dot(self.Xe)).dot(self.Xe.T).dot(self.y)
    else:
      return np.linalg.inv(self.Xen.T.dot(self.Xen)).dot(self.Xen.T).dot(self.y)



  def normalize_vector(self, vector):
    vector_n = []
    for i in range(vector.shape[0]):
      vector_n.append((vector[i] - self.mean_values[i]) / self.std_values[i])

    return np.array(vector_n)

  def cost_fun(self, b = None):
    if b is None: b = self.normal_equation()
    cost = (self.y - (self.Xen.dot(b)))
    cost = cost.T.dot(cost)
    return cost / self.Xen.shape[0]

  def gradient_docent(self, b, a):
    return np.subtract(b, (((a) * (self.Xen.T)).dot(np.subtract(self.Xen.dot(b), self.y))))
