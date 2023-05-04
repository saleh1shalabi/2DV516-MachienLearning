import numpy as np
import matplotlib.pyplot as plt
from A2 import REG

data = np.loadtxt("A2_datasets_2022/GPUbenchmark.csv", delimiter=",", dtype=np.float64)

X = data[:, :6]
y = data[:, 6]


reg = REG(X, y)
###################################
# Task (1) normalizing X
Xn = reg.Xn
###################################


##############################################
# Task (2) Plotting Normalized values
def plot(X, y, title):
  plt.figure(figsize=(10,5))
  plt.suptitle(title)

  for i in range(X.shape[1]):
    plt.subplot(2, 3, i + 1)
    plt.scatter(X[:, i], y)


plot(Xn, y, "normalized values")
plt.show()
###############################################

###############################################
# Task (3) Computing beta and predicting a test

Xe = reg.Xe
beta = reg.normal_equation()


test = np.array([2432, 1607, 1683, 8, 8, 256])
test = reg.normalize_vector(test)
test = np.append(np.array(1), test)

print(f"predicted value of the test using the normal equation = {test.dot(beta)}")


##########################
# Task (4) the cost of beta from normal equation
cost = reg.cost_fun()
print("cost of beta from normal equation=", cost)
##########################

# Task (5)
# (a)

start_beta = np.array([0, 0, 0, 0, 0, 0, 0])
bet = start_beta
alpha = 0.027  # biggest step that doesn't lead to increasing the cost
iter = 1500 # gets the beta from Gradient docent to under 1% diff from the one with normal equation
g_d = []
g_d.append(bet)

for c in range(iter):
  bet = reg.gradient_docent(bet, alpha)
  g_d.append(bet)
  if reg.cost_fun(bet) > reg.cost_fun(g_d[c]):
    print(f"to big steps, iter = {c}, alpha = {alpha}")
    print(f"cost for prev = {reg.cost_fun(g_d[c])}, cost now = {reg.cost_fun(bet)}")
    break

print("alpha:", alpha, "iteration:", iter)
cost = reg.cost_fun(g_d[-1])
print("cost of gradient descent=", cost)



# (b)
print(f"predicted value using beta from gradient descent: {test.dot(g_d[-1])}")
