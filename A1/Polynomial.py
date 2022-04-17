from FileReader import File_Reader
from KNN import KNN
import numpy as np
import matplotlib.pyplot as plt

g = File_Reader("A1_datasets/polynomial200.csv")

data = np.split(g.get_data(), 2)

x = data[1]


knn = KNN(data[0])

plt.figure()
plt.subplot(1,2,1)
plt.title("Train_set")
plt.scatter(data[0][:,0], data[0][:,1], c="b")

plt.subplot(1,2,2)
plt.title("Test_set")
plt.scatter(data[1][:,0], data[1][:,1], c="r")



def regrission(k):
  x_axis = np.linspace(min(knn.coordinate[:, 0]), max(knn.coordinate[:, 0]), 200)
  reg = []

  for r in x_axis:
    t = knn.nearest_by_X(r, k)
    y = 0

    for tt in t:
      row = knn.train_set[tt]
      y += row[1]

    reg.append([r, y / t.shape[0]])
  reg = np.array(reg)

  reg = reg[reg[:, 0].argsort()]
  return reg


def mse_value(k, train_values, test_values):
  x_axis = test_values[:, 0]
  y_axis = test_values[:, 1]

  mse = 0
  # reg = []

  for r in range(x_axis.shape[0]):
    t = knn.nearest_by_X(x_axis[r], k)
    actula = y_axis[r]
    y = 0
    for tt in t:
      row = train_values[tt]
      y += row[1]
    mse += (actula - (y / t.shape[0])) ** 2

  return mse / x_axis.shape[0]


plt.figure(2, figsize=(15, 15))
f = 1
for k in range(1, 12, 2):
  reg = regrission(k)
  mse = mse_value(k, knn.coordinate, knn.coordinate)
  print("MSE for test_set of K =", str(k) + ", " , str(mse_value(k, knn.coordinate, x))[:5])

  plt.subplot(2, 3, f)

  plt.title("ploynomnial_train, k = " + str(k) + ", MSE = " + str(mse)[:5])
  plt.scatter(data[0][:, 0], data[0][:, 1], c="b")

  plt.plot(reg[:, 0], reg[:, 1], c="r")

  f += 1

plt.show()
