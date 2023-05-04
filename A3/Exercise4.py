import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from funcs import *
data = np.loadtxt("Arkiv/bm.csv", delimiter=",", dtype=np.float64)


np.random.shuffle(data) ## shuffled to be good for fit.

pros = 0.50


test_data = data[:round(data.shape[0]*pros)]
X_test, y_test = test_data[:,:test_data.shape[1]-1], test_data[:,test_data.shape[1]-1]
data = data[round(data.shape[0]*pros):,]
X, y= data[:,:data.shape[1]-1], data[:,data.shape[1]-1]


trees = []
for i in range(100):
  g = np.random.choice(range(data.shape[0]), size=data.shape[0], replace=True)
  tr = data[g]
  Xt, yt = tr[:,:tr.shape[1]-1], tr[:,tr.shape[1]-1]
  # trees.append(DecisionTreeRegressor().fit(Xt,yt))
  trees.append(DecisionTreeClassifier().fit(Xt,yt))



## forest error?
predictions = np.zeros(y.shape[0])
for tree in trees:

  predictions += tree.predict(X)


z = np.where(predictions < 50)
o = np.where(predictions >= 50)
predictions[z] = 0
predictions[o] = 1

miss_classi = sum(y != predictions)
print(f"number of miss classified points using the forest = {miss_classi}")

print(f"error rate of forest= {miss_classi/y.size}")


# Average error?
sums = []

for tree in trees:
  pred = tree.predict(X)
  sums.append(sum(y!= pred) / y.size)


print(f"average of error rate = {sum(sums)/len(sums)}")




exit()

from time import time
t_s = time()

grids = []

for tree in trees:
  grids.append(dec_boundry(1000, tree,X))


print(time()-t_s)

for i in range(len(grids)):
  if i == 99: break
  plt.subplot(10,10,i+1)
  plt.contour(grids[i])


plt.subplot(10,10,100)
f = dec_boundry(1000, trees, X, trees=True)
plt.contour(f, colors=["black"])



text = """
          The result that the models give was over the expected one
          since it had random train set, and no params at all was used for the trees
          the benefits of it that it actually had a good accuracy for the data
          and since the voting the values predicted is more true than what was thought
          so it could work well when the data have unknown source, the trees gives an easy
          way to classify the data. 
          the downside that since we need a large number of trees to get more accurate predictions
          this could give lack in both preference and memory.
          
"""

plt.show()
