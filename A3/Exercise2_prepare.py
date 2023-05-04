from MNIST_FileReader import get_train_set
from funcs import *
from time import time
import pickle


"""
      Don't run this file since the outcome of it is saved only run Exercise2 file to see result

"""





train_size = 60000 # the train size to get

print("getting data!")
data= get_train_set(train_size)
print("data collected!")


np.random.shuffle(data) ## shuffled to be good for fit.

X, y = data[:,:data.shape[1]-1], data[:,data.shape[1]-1]

## deviding to get validation set
pros = 0.30 # 30% of the set is used as validation set
y_test = y[:round(y.shape[0]*pros),]
y = y[round(y.shape[0]*pros):,]
X_test = X[:round(X.shape[0]*pros),]
X = X[round(X.shape[0]*pros):,]



one_vs_one_file = "one_vs_one3.pkl"
one_vs_all_file = "one_vs_all3.pkl"





## OBS:: the fit progress of the following lists will take long time
## it'v been tested with both gridSearchCV and my on implemntiation
## my methods works much faster since threads and gives the same result
## one thing to mintion is deppining on the size of set C and gamma could change
## that is since the best C and gamma for 1000 sample is not the same as for 10000 or 60000
## when using small size for the learning it is hard to get over 95% accuracy
## the bigger it is the easier it gets to get accuracy up to 95% or higher



#######################################################################################################
# these numbers were used in the search after the best params but since it take long time             #
# I have run it once with the complete set, and it took about 4h.                                     #
# since the long time it takes, the best model was saved into a pkl file using the pickle lib         #
# if the 'scale' value of gamma is not in the list the best fit param of gamma was 1.9e-7             #
# gammas = [1, 0.1, 0.03, 0.5, 0.0001, 1.9e-7, "scale"]                                               #
# C = [1,2, 5, 10, 100, 1000, 1000000]                                                                #
#######################################################################################################

###################################
# best params for complete set    #
C = [100]                         #
gammas = ["scale"]                #
###################################


## using the grid search takes much longer time than the methods
## I imolemented, thats why i choose to go with validation set to choose the
## best model params.

# gridsearch :
# svc = SVC(kernel='rbf')
# t_s = time()
# param_grid = [{'C': C, 'gamma': gammas}]
# clf = GridSearchCV(svc, param_grid)
# clf.fit(X,y)
# print(f"time grid search = {time()-t_s}")
# # print(clf.get_params(deep=True))
# print(clf.best_params_)
# exit()

# ############################################################################################

l, clfs , p = fitter_maker(C,gammas, [])  ## only the clf what is needed
# the l is for linear and p is for polynomial. they are not needed here
l, p = None, None

print("fitting one vs one....")

t_s = time()

estimators_fitter(clfs, X, y)

print(f"Time for fitting one vs one models = {time()-t_s}")

print("fitted")

print("choosing the best estimator and finding out params")
ind , score = fitter_chooser(clfs, X_test, y_test)
params = clfs[ind].get_params()
print(f"C = {params['C']}, gamma = {params['gamma']}")


print("saving the models to file...")
with open(one_vs_one_file, 'wb') as file:
  pickle.dump(clfs[ind], file)
print("models saved")

############################################################################################

## biniraizing the set
sets = multi_to_bi(y)

# creating 10 models
models= []
for i in range(len(sets)):
  models.append(SVC(C = params['C'], gamma=params['gamma'], probability=True))

print("fitting all models of One Vs All.....")

t_s = time()

threads = []
for m in range(len(models)):
  t = Thread(target=models[m].fit, args=(X,sets[m]))
  t.start()
  threads.append(t)
for t in threads:
  t.join()

print(f"Time for fitting one vs all models = {time()-t_s}")

print("fitted")


print("saving the models to file...")
with open(one_vs_all_file, 'wb') as file:
  pickle.dump(models, file)
print("models saved!")




