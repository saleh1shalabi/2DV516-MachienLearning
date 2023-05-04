import numpy as np



"""
  The classes I used to solve the assignment
  they are very much the same with same with some little changes in each to best for the task
  it is used for
  
  there is a much better why ti fix them doing an abstract class to reduce all code duplication
  I did not have the time to fix it better, and I assumed it is not necessary since there where
  no motioning of code quality need.
  
"""

class REG:
  """
  linear regression class
  """
  X = None
  y = None
  Xn = None
  Xe = None
  Xen = None
  mean_values = []
  std_values = []

  def __init__(self, arr, y, p = 0):
    self.X = arr
    self.y = y
    ones = np.ones(self.X.shape[0])

    for i in range(self.X.shape[1]):
      self.mean_values.append(np.mean(self.X[:, i]))
      self.std_values.append(np.std(self.X[:, i]))

    self.__normalize_values()
    self.Xe = np.column_stack((ones,self.X))
    self.Xen = np.column_stack((ones, self.Xn))


  def __normalize_values(self):
    self.Xn = []
    for i in range(self.X.shape[1]):
      g = (self.X[:, i] - self.mean_values[i]) / self.std_values[i]
      self.Xn.append(np.array(g).T)
    self.Xn = np.array(self.Xn).T


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
    """
    normalize a vector to on the set
    :param vector: the vector to normalize
    :return:
    normalized values of the vector
    """
    vector_n = []
    for i in range(vector.shape[0]):
      vector_n.append((vector[i] - self.mean_values[i]) / self.std_values[i])

    return np.array(vector_n)

  def cost_fun(self, b = None):
    """
    the cost function of the class
    :param b: beta: when doing gradient descent the b param is needed
    :return:
    the MSE
    """
    if b is None: b = self.normal_equation()
    cost = (self.y - (self.Xen.dot(b)))
    cost = cost.T.dot(cost)
    return cost / self.Xen.shape[0]

  def gradient_docent(self, b, a):
    return np.subtract(b, (((a) * (self.Xen.T)).dot(np.subtract(self.Xen.dot(b), self.y))))



class POL_REG:
  """
  the class for polynomial regression
  """
  degree = None
  X = None
  y = None
  Xn = None
  Xen = None
  mean_values = None
  std_values = None
  beta = None

  def __init__(self, arr, y, degree = 1):

    if degree < 1:
      raise ValueError("degree must be at least one")
    self.X = arr
    self.y = y
    self.degree = degree
    ones = np.ones(self.X.shape[0])

    self.mean_values = np.mean(self.X)
    self.std_values = np.std(self.X)

    self.__normalize_values()

    self.Xn = self.__deg(self.Xn)

    self.Xe = np.column_stack((ones,self.X))
    self.Xen = np.column_stack((ones, self.Xn))
    self.beta = self.normal_equation()

  def __deg(self, X):
    """
    extends the X to the degree passed in making the class
    """
    if self.degree == 1:
      pass

    if self.degree != 1:
      c1 = X
      degs = self.degree
      for i in range(degs):
        if i == 0: continue
        else:
          X = np.column_stack((X, c1**(i+1)))

    return X

  def __normalize_values(self):
    self.Xn = []
    g = np.subtract(self.X, self.mean_values) / self.std_values
    self.Xn.append(np.array(g).T)
    self.Xn = np.array(self.Xn).T


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
    vector_n.append((vector - self.mean_values) / self.std_values)
    v = self.__deg(np.array(vector_n))
    return np.array(v)

  def cost_fun(self, b = None):
    if b is None: b = self.beta
    cost = (self.y - (self.Xen.dot(b)))
    cost = cost.T.dot(cost)
    return cost / self.Xen.shape[0]


  def predict(self,vector, price):
    """
    used to predict hte house value in the task
    :param vector:
    :param price:
    :return:
    """
    ind = vector.dot(self.beta)
    p = (ind / self.mean_values) + 1
    p*=price
    return p




class LOG_REG:
  """
  the linear logistic regression class
  """
  X = None
  y = None
  Xn = None
  Xe = None
  Xen = None
  mean_values = []
  std_values = []
  beta = None
  def __init__(self, arr, y):

    self.X = arr
    self.y = y
    ones = np.ones(self.X.shape[0])

    for i in range(self.X.shape[1]):
      self.mean_values.append(np.mean(self.X[:, i]))
      self.std_values.append(np.std(self.X[:, i]))

    self.__normalize_values()
    self.Xe = np.column_stack((ones,self.X))
    self.Xen = np.column_stack((ones, self.Xn))



  def __normalize_values(self):
    self.Xn = []
    for i in range(self.X.shape[1]):
      g = (self.X[:, i] - self.mean_values[i]) / self.std_values[i]
      self.Xn.append(np.array(g).T)
    self.Xn = np.array(self.Xn).T



  def get_norm_set(self, e = True):
    """
    called to get the normalized dataset.
    :return:
      normalized data
    """
    if not e: return self.Xn
    else : return self.Xen


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
    cost = self.y.T.dot(np.log(self.sigmoid(self.Xen.dot(b))))
    cost += (1 - self.y).T.dot(np.log(1 - self.sigmoid(self.Xen.dot(b))))
    return -(cost / self.Xen.shape[0])



  def sigmoid(self, X):
    return (np.e ** X) / ((np.e ** X) + 1)


  def gradient_docent(self, b, a):
    s = self.sigmoid(self.Xen.dot(b))
    s -=  self.y
    g = b - a / (self.Xen.shape[0]) * (self.Xen.T).dot(s)
    return g



  def dec_boundry(self, grid_size, b):
    """
    makes the decision boundry for the set of a given beta and given grid size
    :param grid_size:
    :param b:
    :return:
    """
    min_value_x, max_value_x = min(self.Xe[:, 1]) - 0.5, max(self.Xe[:, 1]) + 0.5
    min_value_y, max_value_y = min(self.Xe[:, 2]) - 0.5, max(self.Xe[:, 2]) + 0.5

    x_axis = np.linspace(min_value_x, max_value_x, grid_size)
    y_axis = np.linspace(min_value_y, max_value_y, grid_size)
    grid = np.zeros(shape=(len(x_axis), len(y_axis)))
    for row in range(grid_size):
      for column in range(grid_size):

        grid[row, column] = round(self.sigmoid(np.array( [1, x_axis[row], y_axis[column]]).dot(b)))
    return grid





class NON_LOG_REG:

  """
  Non-linear logistic regression
  """

  X = None
  y = None
  Xn = None
  Xe = None
  Xen = None
  mean_values = []
  std_values = []
  beta = None
  degree = 0

  def __init__(self, arr, y, degree = 0):
    self.degree = degree
    if degree > 1:
      self.X = self.mapFeature(arr[:,0], arr[:,1], degree)
    else:
      self.X = arr
    self.y = y
    ones = np.ones(self.X.shape[0])

    for i in range(self.X.shape[1]):
      self.mean_values.append(np.mean(self.X[:, i]))
      self.std_values.append(np.std(self.X[:, i]))

    # self.__normalize_values()
    self.Xn = self.X

    self.Xe = np.column_stack((ones,self.X))
    self.Xen = np.column_stack((ones, self.Xn))



  def __normalize_values(self):
    self.Xn = []
    for i in range(self.X.shape[1]):
      g = (self.X[:, i] - self.mean_values[i]) / self.std_values[i]
      self.Xn.append(np.array(g).T)
    self.Xn = np.array(self.Xn).T


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
    cost = self.y.T.dot(np.log(self.sigmoid(self.Xen.dot(b))))
    cost += (1 - self.y).T.dot(np.log(1 - self.sigmoid(self.Xen.dot(b))))
    return -(cost / self.Xen.shape[0])



  def sigmoid(self, X):
    return (np.e ** X) / ((np.e ** X) + 1)


  def gradient_docent(self, b, a):
    s = self.sigmoid(self.Xen.dot(b))
    s -=  self.y
    g = b - a / (self.Xen.shape[0]) * (self.Xen.T).dot(s)
    return g



  def dec_boundry(self, grid_size, b):

    min_value_x, max_value_x = min(self.X[:, 0]) - 0.5, max(self.X[:, 0]) + 0.5
    min_value_y, max_value_y = min(self.X[:, 1]) - 0.5, max(self.X[:, 1]) + 0.5

    x_axis = np.linspace(min_value_x, max_value_x, grid_size)
    y_axis = np.linspace(min_value_y, max_value_y, grid_size)
    grid = np.zeros(shape=(len(x_axis), len(y_axis)))
    for row in range(grid_size):
      for column in range(grid_size):
        z = self.mapFeature(x_axis[row],y_axis[column], self.degree)
        grid[row, column] = round(self.sigmoid(np.append(1,z).dot(b)))
    return grid


  def mapFeature(self,X1,X2,degree):
    """
    little changed but almost the same as from the lecture slide
    :param X1:
    :param X2:
    :param degree:
    :return:
    """
    X = np.column_stack((X1,X2))
    for i in range(2,degree+1):
      for j in range(0,i+1):
        new = X1**(i-j)*X2**j
        X = np.column_stack((X,new))
    return X


