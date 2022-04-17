import numpy as np



class KNN():
  """The class used to solve the problem
  takes in the numpy array to work with from the File_Reader

  """
  train_set = None
  coordinate = None
  result_values = None
  min_value_x = None
  max_value_x = None
  min_value_y = None
  max_value_y = None

  def __init__(self, train_set):
    self.train_set = train_set
    self.coordinate = train_set[:, :2]
    self.min_value_x = min(self.coordinate[:, 0])
    self.max_value_x = max(self.coordinate[:, 0])
    self.min_value_y = min(self.coordinate[:, 1])
    self.max_value_y = max(self.coordinate[:, 1])

  def eculidean_distance(self, row, point):
    """calculates Eculidean distance

    Args:
        row (coordinate point (x, y)): _description_
        point (coordinate point (x, y)): _description_

    Returns:
        float: the dictanse masured.
    """
    return np.linalg.norm(row - point)

  def manhaten_distance(self, row, point):
    """calculates Manhaten distance
    Args:
        row (coordinate point (x, y)): _description_
        point (coordinate point (x, y)): _description_

    Returns:
        float: the dictanse masured.
    """
    return np.sum(np.abs(row - point))

  def get_neighbors(self, point, algo=0):
    """calculate the distance of each point of the set to the given point
    Args:
        point (coordinate point (x, y)): _description_
        algo (int): 0 or 1, 0 is defualt to use eculadien_distance 
                    1 to use manhaten_distance
    Returns:
        numpy array: contains the distance to each point in a0 and if it was 0 or 1 in a1
    """

    neighbors_list = []

    for row in self.train_set:
      if algo == 0:

        if np.array_equal(row[:2], point):
          continue
        else:
          neighbors_list.append([self.eculidean_distance(row[:2], point), row[-1]])
      else:
        if np.array_equal(row[:2], point):
          continue
        else:
          neighbors_list.append([self.manhaten_distance(row[:2], point), row[-1]])
    neighbors_list = np.array(neighbors_list)
    neighbors_list = neighbors_list[neighbors_list[:, 0].argsort()]

    return neighbors_list

  def predict(self, k_value, point, algo=0):
    """take a point to give a predicted value of pass or fail compared to the set
    Args:
        k_value (int): _description_
        point (coordinate point (x, y)): _description_
        algo (int, optional): Defaults to 0. 1 to use manhaten_distance
    Returns:
        int: value of 0 or 1 (pass or fail) of point.
    """
    k = 0
    ok = 0
    fail = 0
    neighbors = self.get_neighbors(point)

    while k != k_value:
      if neighbors[k][1] == 1.:
        ok += 1
      else:
        fail += 1
      k += 1
    if ok > fail:
      return 1

    elif fail > ok:
      return 0
    else:
      print("error occurred")
      print("the K-value must be odd")
    exit()

  def training_accuracy_error(self, k_value, a = "",):
    """
    calculates the training error and accuracy
    Args:
      k_value: the neighbors value
      a: "" for accuracy "e" for error rate

    Returns:
      the value of accuracy
    """
    ok = 0
    fail = 0
    for p in self.train_set:
      if self.predict(k_value, p[:2]) == p[2]:
        ok += 1
      else: fail += 1
    if a == "" or a == "e":
      if a == "e":
        return fail/self.train_set.shape[0]
      else:
        return ok/self.train_set.shape[0]
    else:
      print("parameter invalid")
      exit()

  def dec_boundry(self, grid_size, k_value):
    """ calculate the grid array to use in imshow.

    Args:
        grid_size (int): what size of grid 
        k_value (int): the numper of neighbors

    Returns:
        numpy_array: return the array to use in imshow
    """
    x_axis = np.linspace(self.min_value_x, self.max_value_x, grid_size)
    y_axis = np.linspace(self.min_value_y, self.max_value_y, grid_size)
    grid = np.zeros(shape=(len(x_axis), len(y_axis)))
    for row in range(grid_size):
      for column in range(grid_size):

        # grid[row, column] = self.predict(k_value,[x_axis[row], y_axis[column]])

        ## a faster way to calculate grid.
        ## use when grid is larger than 100x100
        ## it finds only the near_neighbors to calculate the distance to.
        ## instead of calculating the distance to each point of the set.
        grid[row, column] = self.dec_pred(k_value, [x_axis[row], y_axis[column]])

    return grid


  def dec_pred(self, k_value, point):
    """
    this method is used internally to calculate the decision boundary grid
    Args:
      k_value: neighbors number
      point: point on grid
    Returns:  the value to be assigned for point on the grid of 1 or 0
    """
    k = 0
    ok = 0
    fail = 0
    neighbors = self.dec_grid(point, k_value)
    while k != k_value:
      if neighbors[k][1] == 1.:
        ok += 1
      else:
        fail += 1
      k += 1
    if ok > fail:
      return 1
    elif fail > ok:
      return 0
    else:
      print("error occurred")
      print("the K-value must be odd")
    exit()

  def dec_grid(self, point, k_value):
    """
    used internally to find distances to the neighbors
    Args:
      point: poit of grid
      k_value: neighbors

    """
    neighbors_list = []
    point = np.array(point)

    v = self.get_near_values(point, k_value)

    for row in v:
      t = self.train_set[row]
      if np.array_equal(t[:2], point):
        continue
      else:
        neighbors_list.append([self.eculidean_distance(t[:2], point), t[-1]])
    neighbors_list = np.array(neighbors_list)
    neighbors_list = neighbors_list[neighbors_list[:, 0].argsort()]

    return neighbors_list

  def get_near_values(self, point, k_value):
    """
    Find the absolute nearest neighbors of a point before calculating distances.
    Args:
      point: the point coordinates
      k_value: neighbors number
    Returns:
        an array with absolute the nearest points
    """

    x = self.coordinate[:, 0]
    y = self.coordinate[:, 1]
    marg_value = 0.1

    # this command creates an array of the neighbors for the point
    # it works making a circle around the point and looks inside it for neighbors
    v = np.where(np.logical_and(np.logical_and(x < point[0] + marg_value, x > point[0] - marg_value),
                                np.logical_and(y < point[1] + marg_value, y > point[1] - marg_value)))[0]

    #  if the number of neighbors found in the circle around the point was less
    # than the number in parameter it increases the circle radius  and looks again.
    while v.shape[0] < k_value:
      marg_value += 0.1

      v = np.where(np.logical_and(np.logical_and(x < point[0] + marg_value, x > point[0] - marg_value),
                                  np.logical_and(y < point[1] + marg_value, y > point[1] - marg_value)))[0]

    return v

  def nearest_by_X(self, x_value, k_value, plus_value=0.0001):
    """
    Works as the func get_near_values, but it finds the nearst only in the X axis
    it means a value can be found at the same
    Returns:
      array of the nearest points on the X axis
    """
    x = self.coordinate[:, 0]
    marg_value = 0.0
    array = np.where(np.logical_and(x <= x_value + marg_value, x >= x_value - marg_value))[0]
    while array.shape[0] < k_value:
      marg_value += plus_value
      array = np.where(np.logical_and(x <= x_value + marg_value, x >= x_value - marg_value))[0]
    return array
