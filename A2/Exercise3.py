import numpy as np
from A2 import LOG_REG
import matplotlib.pyplot as plt


data = np.loadtxt("A2_datasets_2022/breast_cancer.csv", delimiter=",", dtype=np.float64)

np.random.shuffle(data)

X = data[:,:9]
y = np.floor(data[:,-1] / 4) # converting to 0 and 1
y_test = y[:round(y.shape[0]*0.30),]
y = y[round(y.shape[0]*0.30):,]
X_test = X[:round(X.shape[0]*0.30),] # 30% of the rows to be test set
X = X[round(X.shape[0]*0.30):,] # the rest of the rows to be the train set

## I tried 20% of the set as test and 30% since there is no noticeable different in the accurasy
## I choose to go with 30% for test

reg = LOG_REG(X,y)

Xen = reg.get_norm_set()


## normalizing the test set
for i in range(X_test.shape[0]):
  X_test[i] = reg.normalize_vector(X_test[i])

X_test = np.column_stack((np.ones(X_test.shape[0]), X_test))


#################################################################################
# from A_N_func import fun
# ff = fun(reg)
# alpha, i,beta,g = ff.find(0.01) ## finds the best alpha and iterations
# print(alpha,i)
#################################################################################
## using the commented code above gives avreage alpha of 1 for multiple runs
## the alpha is different each time since the shuffling
## the iter in avreage little less than 1000


              ######################
            #### Gradient descent ####
#################################################
start_beta = np.zeros(reg.Xen.shape[1])         #
a = 1                                           #
iter = 1000                                     #
g = [start_beta]                                #
for i in range(iter):                           #
  g.append(reg.gradient_docent(g[i],a))         #
#################################################
reg.beta = g[-1]

## cost of following is always in range ca 0.058 - 0.085
## and not always the same since the shuffling
## the accurasy of training is always over 96% in all tests I've done
## the least accurasy I got for the testset is 94% and highst was 99.02%


print("alpha value:", str(a)[:4], "|| iterations: ",iter)

print("cost", reg.cost_fun(g[-1]))


yt = np.arange(0,iter+1, 1).tolist()
for c in range(len(g)):
  g[c] = reg.cost_fun(g[c])

plt.plot(yt,g)

er = 0

for i in range(Xen.shape[0]):
  if (round(reg.sigmoid(Xen[i].dot(reg.beta)))) != int(y[i]): er+=1

print("train errors:", er)
print("train accuracy =", str(round(((y.shape[0] - er) / y.shape[0]),6 )*100)[:5] + "%")
print("train error rate=", str( round( (er/y.shape[0]),6 )*100)[:5] + "%")

er = 0
for i in range(X_test.shape[0]):
  if (round(reg.sigmoid(X_test[i].dot(reg.beta)))) != int(y_test[i]): er+=1

print("="*20)
print("test error rate:", er)
print("test accuracy =", str(round( ((y_test.shape[0] - er) / y_test.shape[0]),6)*100)[:5] + "%")
print("test error rate=", str(round((er/y_test.shape[0]), 6)*100)[:5] + "%")

plt.show()


"""
the repeated runs will result in qualitatively the same result
they depends a bit on how the complete set is split since the more train data the model have 
the better result it will give
since all data is near the result was expected
"""
