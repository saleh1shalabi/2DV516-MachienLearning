import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from A2 import NON_LOG_REG
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict


print("This task takes time due all D_B to count!!")

data = np.loadtxt("A2_datasets_2022/microchips.csv", delimiter=",", dtype=np.float64)

X_0, y = data[:,:2], data[:,2]

sets = []
for i in range(1,10):
  sets.append(NON_LOG_REG(X_0,y,i).Xen)

cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])


def dec_boundry(grid_size, degree, skl):

  min_value_x, max_value_x = min(X_0[:, 0]) - 0.5, max(X_0[:, 0]) + 0.5
  min_value_y, max_value_y = min(X_0[:, 1]) - 0.5, max(X_0[:, 1]) + 0.5

  x_axis = np.linspace(min_value_x, max_value_x, grid_size)
  y_axis = np.linspace(min_value_y, max_value_y, grid_size)
  grid = np.zeros(shape=(len(x_axis), len(y_axis)))
  for row in range(grid_size):
    for column in range(grid_size):
      z = mapFeature(x_axis[row],y_axis[column], degree)
      z = np.column_stack((1,z))
      grid[row, column] = skl.predict(z)
  return grid


def mapFeature(X1,X2,degree):
  X = np.column_stack((X1,X2))
  for i in range(2,degree+1):
    for j in range(0,i+1):
      new = X1**(i-j)*X2**j
      X = np.column_stack((X,new))
  return X


def train_error(skl, X):
  er = 0
  t = skl.predict(X)
  for i in range(t.shape[0]):
    if (t[i]) != int(y[i]): er+=1
  return er


############
#  Task 1  #
############
def task(skl, sets):
  plt.figure()
  for i in range(1,10):
    plt.subplot(3,3, i)
    plt.scatter(X_0[:,0], X_0[:,1], c=y, marker=".", cmap=cmap_bold)
    skl.fit(sets[i-1], y,)
    plt.title(f"degree {i}, train errors {train_error(skl, sets[i-1])}")
    plt.imshow(dec_boundry(100,i, skl).T,origin='lower',
                     extent=(min(X_0[:, 0])- 0.5, max(X_0[:, 0]) + 0.5,
                             min(X_0[:, 1])- 0.5, max(X_0[:, 1])+ 0.5),
    cmap=cmap_light )




skl = LogisticRegression(C=10000, max_iter=1000)
task(skl, sets)
plt.suptitle("C = 10000")

############
#  Task 2  #
############
skl_ = LogisticRegression(C=1)
task(skl_, sets)
plt.suptitle("C = 1")



############
#  Task 3  #
############

regularized = []
un_regularized = []
errs_r = []
errs_u = []

for i in sets:

  regularized.append(cross_val_predict(skl, i, y))
  un_regularized.append(cross_val_predict(skl_, i, y))

for i in range(len(regularized)):
  un = un_regularized[i]
  r = regularized[i]
  er = 0
  uner = 0
  for j in range(y.shape[0]):
    if un[j] != y[j]: uner+=1
    if r[j] != y[j]: er +=1
  errs_r.append(er)
  errs_u.append(uner)


degs = [i for i in range(1,10) ]

plt.figure()
plt.plot(degs,errs_r, label="regularized")
plt.plot(degs,errs_u, label="un-regularized")
plt.legend()


print("the low value of param C makes the model less flexible since it regularizes the model to not be over fitted")
print(" and the opposite is True ")
plt.show()



