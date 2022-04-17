from numba import jit, cuda
import numpy as np

print(np.__version__)
@jit(target= "cuda")
class KNN_MNIST():
  """
    This class is used to be the number classifier.
    To get higher performance the @param sample is used
  """
  train_data = None
  samples = 0
  q_s = None

  def __init__(self, train_data, samples):
    """
      defines the classifier object.
    Args:
      train_data: is the data_set to be used.
      samples: is the number of samples the test is compared to.
    """

    self.train_data = train_data
    self.samples = samples




  def on_knn(self, test, neighbors):
    """
    calculates the Euclidean distance of the test array
    against a number of sample from each number of the train_set
    find the least "neighbors" values of each number
    sums and take the average of it
    the least value is in the index 0 - 9 which represents the number
    Args:
      test: test grid
      neighbors: count of neighbors

    Returns:
      the predicted value
    """

    distances = []
    t_t = []
    for c in range(10):
      t_t.append([])

    index = 0
    # each array is an array containing arrays of one number
    for array in self.train_data[:, 0]:
      distances = []
      # creates an array of random number between 0 and samples
      rand_pics = np.random.choice(self.samples, size=self.samples, replace=False)
      # from the array the grids in positions in array will be taken
      rand_arrey = array[rand_pics,
                   :]  # this is an array containing samples number of grids which has been randomly chosen

      g = (np.linalg.norm(grid - test) for grid in rand_arrey)
      g = np.sort(np.array(list(g)))

      t_t[index] = g[:neighbors, ]

      for t in t_t:
        distances.append(np.sum(np.array(t) / np.array(t).shape[0]))
      index += 1
    # the index where the least value found is the predicted number
    print("WTF")
    return distances.index(min(distances))
