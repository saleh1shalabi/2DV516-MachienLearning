import numpy as np
import matplotlib.pyplot as plt
import pickle


file_path = "neuralNetwork_complete.pkl"
labels = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

print("reading data...")
data = np.loadtxt("Arkiv/fashion-mnist_train.csv", delimiter=",", dtype=np.float64,skiprows=1)
test = np.loadtxt("Arkiv/fashion-mnist_test.csv", delimiter=",", dtype=np.float64,skiprows=1)


X = data[:,1:] / 255
y = data[:,0]
X_test = test[:,1:] / 255
y_test = test[:,0]






with open(file_path, "rb") as file:
  nur_net = pickle.load(file)

score_train = nur_net.score(X,y)
score_test = nur_net.score(X_test,y_test)

print(nur_net.get_params())

print("Score of Training set =", score_train)
print("Score of Test set =", score_test)

rand_pics = np.random.choice(range(X.shape[0]), size=16)

k = 1
for i in rand_pics:
  plt.subplot(4, 4, k)
  k+=1
  plt.imshow(X[i].reshape(28,28), cmap="Greys")

  plt.title(labels[int(y[i])])



preds = nur_net.predict(X_test)

matrix = np.zeros((10,10))
for i in range(y_test.size):
  matrix[int(preds[i]), int(y_test[i])] += 1


plt.matshow(matrix)



for (i, j), z in np.ndenumerate(matrix):
  plt.text(j, i, '{:0.0f}'.format(z), ha='center', va='center')


plt.xticks(np.arange(10), labels)
plt.yticks(np.arange(10), labels)

text = """
        We can se that shirt and t-shirt/top get mixed up of the classifier
        and they are the most miss-classified to each others categories
        the shirt was most miss classified to t-shit/top
        the next most miss-classified categories are the coat which was mixed up
        with the pullover, 
        even the shirt was sometimes mixed up with the pullover
        
        
        in conclusion there are 3 categories that often get mixed up
        the t-shirt, shirt and coat
        then comes the pullover
        
        surprising that the sneakers and Ankle boot was not that often mixed up
        
        easiest classified was bags followed by sandal sneakers and ankle boot  
        
"""

print(text)


plt.show()
