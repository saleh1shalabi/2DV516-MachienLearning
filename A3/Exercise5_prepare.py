import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import pickle


"""
      Dont run this file the outcome is saved into the Exercise5 file

"""


print("reading data...")
data = np.loadtxt("Arkiv/fashion-mnist_train.csv", delimiter=",", dtype=np.float64,skiprows=1)


X = data[:,1:]/255 # making the numbers smaller to be faster in performance
y = data[:,0]





#### the gridsearch takes looooong time!!!!!!!
#### the values are saved into the neuralNetwork_complete.pkl file as an object
params = {"solver":["adam", "lbfgs"],
          "alpha":[0.1, 0.01, 0.001],
          "learning_rate":["adaptive"],
          "hidden_layer_sizes":[(400, 150, 56, 28, 14, 7), (100, 56, 28), (56, 28, 14)],
          "max_iter":[200, 350]}

print("Searching for best estimator")
nw = MLPClassifier(verbose=True)
nw = GridSearchCV(nw, params, n_jobs=-1)

nw.fit(X, y)

print(nw.best_params_)


file_name = "neuralNetwork.pkl" ## changed from the one used in Exercise5 file to not by mistake run
with open(file_name, "wb") as file:
  pickle.dump(nw.best_estimator_, file)

