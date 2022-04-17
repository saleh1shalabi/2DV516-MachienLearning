import numpy as np
import matplotlib.pyplot as plt
def normal_equation(X, y):
  return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

def normalize_values(*args):
  values = np.array(*args)
  return ((values - np.mean(values)) / np.std(values)).T



data = np.loadtxt("./A2_datasets_2022/girls_height.csv")

on = np.ones(data.shape[0])
dad = data[:,2]
mom = data[:,1]
girl = data[:,0]

data_xe = np.column_stack((on,mom,dad))
b = normal_equation(data_xe, girl)

xn = np.column_stack((on, normalize_values([mom,dad])))

dad_n = xn[:,2]
mom_n = xn[:,1]



def plot():
  plt.figure()
  plt.subplot(2,2,1)
  plt.title("DAD-girl")
  plt.scatter(dad, girl)

  plt.subplot(2,2,2)
  plt.title("MOM-girl")
  plt.scatter(mom, girl)

  plt.subplot(2,2,3)
  plt.title("NORM-DAD-girl")
  plt.scatter(dad_n, girl)

  plt.subplot(2,2,4)
  plt.title("NORM-MOM-girl")
  plt.scatter(mom_n, girl)

  plt.show()



test = np.array([1,65,70])

print("Test data for mom: 65, dad: 70\n\tpredicted using the regular_data, girl height is:", str(test.dot(b))[:5])

# plot()



norm_girl = np.array([r.dot(b) for r in xn])


bn = normal_equation(xn, norm_girl)

print("Test data for mom: 65, dad: 70\n\tpredicted using the norm_data, girl height is:", str(test.dot(bn))[:5])



def cos_fun(values, b, y):
  f = 1 / values.shape[0]
  d = values.dot(b)
  m = np.subtract(d, y)

  return (f * m.T).dot(m)

print(cos_fun(data_xe, b, girl))


def gradient_docent(b, a, set, y):
  return np.subtract(b, ((a * (set.T)).dot(np.subtract(set.dot(b), y))) )

g = gradient_docent(b, 0.001, data_xe, girl)

print(cos_fun(data_xe, g, girl))
