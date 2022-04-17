from time import time
import matplotlib.pyplot as plt
from FileReader import File_Reader
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

data = File_Reader("./A1_datasets/microchips.csv").get_data()

x = data[:,:2]
y = data[:,2]

# plt.figure()
#
# plt.subplot(1,2,1)
# plt.scatter(x[:,0], x[:,1])


chips = [[-0.3, 1.0], [-0.5, -0.1], [0.6, 0.0]]

knn = KNeighborsClassifier()
knn.fit(x, y)

# predict chips
for i in range(1,8,2):
  knn.n_neighbors = i
  print("K =", i, end="\n\n")
  values = knn.predict(chips)
  for t in range(len(chips)):
    print("chip :", chips[t-1] ,end=" ==> ")
    if values[t-1] == 1:
      print("OK")
    else:
      print("Fail")
  print("================", end="\n\n")



# creating decision boundery
def dec_bound(k, grid_size):
  x_axis = np.linspace(min(x[:,0]), max(x[:,0]), grid_size)
  y_axis = np.linspace(min(x[:,1]), max(x[:,1]), grid_size)
  grid = np.zeros(shape=(len(x_axis), len(y_axis)))
  for row in range(grid_size):
    for column in range(grid_size):
      grid[row, column] = knn.predict(np.array([[x_axis[row], y_axis[column]]]))[0]
  return grid


# plotting
min_x, max_x = min(x[:, 0]), max(x[:, 0])
min_y, max_y = min(x[:, 1]), max(x[:, 1])

f = 1

for k in range(1, 8, 2):
  t_s = time()
  knn.n_neighbors = k
  grid = dec_bound(k, 100)
  t_e = time()
  print("time of decision boundary calculation of k = ", str(k), " : ", str(t_e - t_s), " s")
  plt.subplot(2, 2, f)
  plt.imshow(grid.T, origin='lower',
             extent=(min_x, max_x, min_y, max_y),
             )
  plt.scatter(x[:, 0], x[:, 1],
              c=y,
              cmap="bwr",
              edgecolors='green')
  plt.title("KNN Boundary k = " + str(k))
  f += 1

plt.show()
