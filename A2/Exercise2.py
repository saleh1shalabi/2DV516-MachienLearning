import numpy as np
import matplotlib.pyplot as plt
from A2 import POL_REG

data = np.loadtxt("A2_datasets_2022/housing_price_index.csv", delimiter=",", dtype=np.float64)

x = np.arange(1975, 2018, 1)
y = data[:,1]

house_price = 2.3
regs = []
b = []
sets = []
test = np.array(2022)
t = []

for i in range(4):
  print("="*25, end="\n\n")
  regs.append(POL_REG(x, y, i+1))
  b.append(regs[i].beta)
  sets.append(regs[i].Xen)
  t.append(regs[i].normalize_vector(test))
  t[i] = np.append(np.array(1),t[i])
  print("*"*12)
  print(f"* Degree {i+1} *")
  print("*"*12)
  print(f"predicts index:", int(t[i].dot(b[i]).round()))
  print(f"The cost is:", int(regs[i].cost_fun(b[i]).round()))
  print(f"Predicted house value is:", regs[i].predict(t[i], house_price).round(2))


plt.figure()
plt.scatter(x, y)


plt.figure()
for c in range(4):
  plt.subplot(2,2,c+1)
  plt.scatter(x, y)
  plt.plot(x, sets[c].dot(b[c]), "red")


print("the best degree is 4 since it has the least cost and the value predicted is what I think realistic")
plt.show()



