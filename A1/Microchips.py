from FileReader import File_Reader
from KNN import KNN
import matplotlib.pyplot as plt
from time import time

f = File_Reader("./A1_datasets/microchips.csv")

knn = KNN(f.get_data())

chips = [[-0.3, 1.0], [-0.5, -0.1], [0.6, 0.0]]

### plot the points.
plt.figure(1)
plt.scatter(knn.coordinate[:,0], knn.coordinate[:,1], c = knn.train_set[:,2])

## predicts chips
for k in range(1, 8, 2):

  print("=" * 25)
  print("K = ", str(k) + "\n")
  print("chip1: --> ", end="")
  if knn.predict(k, chips[0]) == 1:
    print(str(chips[0]), " OK")
  else:
    print(str(chips[0]), " Fail")

  print("chip2: --> ", end="")
  if knn.predict(k, chips[1]) == 1:
    print(str(chips[1]), " OK")
  else:
    print(str(chips[1]), " Fail")

  print("chip3: --> ", end="")
  if knn.predict(k, chips[2]) == 1:
    print(str(chips[2]), " OK")
  else:
    print(str(chips[2]), " Fail")
  print("=" * 25)


## decision boundry plots
f = 1
plt.figure(figsize=(15, 15))
g = 100

for k in range(1, 8, 2):
  t_s = time()
  grid = knn.dec_boundry(100, k)
  t_e = time()
  print("time of decision boundary calculation of k = ", str(k), " : ", str(t_e - t_s), " s")

  plt.subplot(2, 2, f)
  plt.imshow(grid.T, origin='lower',
             extent=(knn.min_value_x, knn.max_value_x, knn.min_value_y, knn.max_value_y),
             )
  plt.scatter(knn.coordinate[:, 0], knn.coordinate[:, 1],
              c=knn.train_set[:, 2],
              cmap="bwr",
              edgecolors='green')
  plt.title("KNN Boundary k = " + str(k) + ", Error rate = " + str(knn.training_accuracy_error(k,"e"))[:5] +
            "\naccuracy = " + str(knn.training_accuracy_error(k))[:5])
  f += 1

plt.show()
