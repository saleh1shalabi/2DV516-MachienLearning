from concurrent.futures import ThreadPoolExecutor
from time import time, sleep
import KNN_MNIST as CK
from MNIST_FileReader import get_train_set, get_test_data


"""
letting each grid of the test be compared to each grid of the training_set
takes a very vary long time, about 6H which is not good for the preforming
using the samples make the performance much faste.       




I had tested many times with different factors of:
 neighbors values : 1 ,3 ,5 ,7      the most accurate was k = 3
 samples values: 100, 150, 250, 300, 500, 1k
 train_size : 1k, 10k, 60k
 
the best results values was for samples = 300
neighbors = 3 
accuracy is 89 - 95 %
the time it takes to run the 60000:10000 is about 18-20 mints on my pc
 which is reasonable compared to 6H


ofcourse running against larger samples gives better accuracy
but preforms much slower. 
"""



train_size = 10000 # the train size to get
test_size = 10 # the test size to get
samples = 1  # the number of samples against each number in the train_set against each test


print("Getting data set!")
data = get_train_set(train_size)
print("Got data as arrays!")
print("*"*20)



print("Getting data test!")
test_set = get_test_data(test_size)
tes_lab = get_test_data(test_size, True)

print("Got test data as arrays!")
print("*"*20)


clf = CK.KNN_MNIST(data, samples)

y = 0
n = 0

print("start test")
print("*"*20)


t_s = time()





l = []
executor = ThreadPoolExecutor(10)

res = []
for g in range(test_size):

  if g in [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000, 9000]:
    print(g)

  if l[g].result() == tes_lab[g]:
    y+=1
  else:
    n+=1



print("Test completed")
print("*"*20, end="\n\n")

print("Test: train_set", str(train_size), ": test_set", str(test_size))
print("Time taken is: ", str(time() - t_s) + "s")
print("tested each pic of the test set against", str(samples), "different pics of each number in training set")
print("accuracy: ", str(y / test_size), "error_rate: ", str(n / test_size))
print("out of ", str(test_size), " predicted right: ", str(y))
print("out of ", str(test_size), " predicted wrong: ", str(n))
