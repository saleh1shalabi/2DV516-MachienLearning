import pickle


from MNIST_FileReader import get_train_set, get_test_data
from funcs import *

one_vs_one = "one_vs_one.pkl"
one_vs_all = "one_vs_all.pkl"


train_size = 60000
test_size = 10000 # the test size to get



print("Reading data...")
data= get_train_set(train_size)
test_set = get_test_data(test_size)
print("Data collected!")

np.random.shuffle(data)
X, y = data[:,:data.shape[1]-1], data[:,data.shape[1]-1]


pros = 0.20
y_test = y[:round(y.shape[0]*pros),]
y = y[round(y.shape[0]*pros):,]
X_test = X[:round(X.shape[0]*pros),]
X = X[round(X.shape[0]*pros):,]

test_X, test_y = test_set[:,:test_set.shape[1]-1], test_set[:,test_set.shape[1]-1]


## reading models from files!
print("importing models from files...")

with open(one_vs_one, 'rb') as file:
  clf = pickle.load(file)

with open(one_vs_all, 'rb') as file:
  models = pickle.load(file)

print("models imported!")


params = clf.get_params()
print(f"C = {params['C']}, gamma = {params['gamma']}")





## validation set accuracy

y_validation_one_vs_one_predicted = clf.predict(X_test)
y_validation_one_vs_all_predicted = one_vs_all_predict_proba(models, X_test)

one_vs_one_score = sum(y_validation_one_vs_one_predicted == y_test)*100
one_vs_all_score = sum(y_validation_one_vs_all_predicted == y_test)*100

print(f"(own implemented) One Vs All accuracy on random validation set= {str(one_vs_all_score/y_test.size)[:6]}%")
print(f"(sklearn) One Vs One accuracy on random validation set= {str(one_vs_one_score/y_test.size)[:6]}%")



### on Test set

y_test_one_vs_one_predicted = clf.predict(test_X)
y_test_one_vs_all_predicted = one_vs_all_predict_proba(models, test_X)

one_vs_one_score = sum(y_test_one_vs_one_predicted == test_y)*100
one_vs_all_score = sum(y_test_one_vs_all_predicted == test_y)*100


print(f"(own implemented) One Vs All accuracy of test set= {str(one_vs_all_score/test_y.size)[:6]}%")
print(f"(sklearn) One Vs One accuracy of test set= {str(one_vs_one_score/test_y.size)[:6]}%")



matrix = np.zeros((10,10))
for i in range(test_y.size):
  matrix[y_test_one_vs_one_predicted[i], test_y[i]] += 1

plt.matshow(matrix)
plt.suptitle("One_VS_One")

for (i, j), z in np.ndenumerate(matrix):
  plt.text(j, i, '{:0.0f}'.format(z), ha='center', va='center')


matrix = np.zeros((10,10))
for i in range(test_y.size):
  matrix[y_test_one_vs_all_predicted[i], test_y[i]] += 1


plt.matshow(matrix)
plt.suptitle("One_VS_All")


for (i, j), z in np.ndenumerate(matrix):
  plt.text(j, i, '{:0.0f}'.format(z), ha='center', va='center')


plt.show()
